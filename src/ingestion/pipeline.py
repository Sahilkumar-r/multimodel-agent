"""Ingestion pipeline.

Ingestion only does work when something changed:

* first run / empty Pinecone namespace   -> ingest everything
* ingestion config changed               -> clear and re-ingest everything
* a file is new / modified / deleted     -> ingest, replace or delete only that file
* nothing changed                        -> no parsing, no embedding, no network writes
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.config import settings
from src.ingestion.chunking import HybridChunker
from src.ingestion.manifest import Manifest, config_fingerprint
from src.ingestion.parsers import ParseError, ParsedImage, classify, parse_document, parse_image
from src.prompts import IMAGE_SUMMARY_SYSTEM
from src.retrieval.sparse import to_corpus_item
from src.utils import (
    embedding_hash,
    fit_metadata,
    make_id,
    now_iso,
    sha256_file,
    truncate,
)

log = logging.getLogger(__name__)

EMPTY_GRACE_SECONDS = 180  # Pinecone stats are eventually consistent right after an upsert


@dataclass
class FileInfo:
    rel: str
    path: Path
    sha256: str
    size: int
    mtime_ns: int


@dataclass
class IngestPlan:
    reason: str = "up to date"
    clear_first: bool = False
    to_add: dict[str, FileInfo] = field(default_factory=dict)
    to_update: dict[str, FileInfo] = field(default_factory=dict)
    to_delete: list[str] = field(default_factory=list)
    unchanged: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def needs_ingestion(self) -> bool:
        return bool(self.clear_first or self.to_add or self.to_update or self.to_delete)


@dataclass
class IngestReport:
    reason: str = ""
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    vectors: int = 0
    seconds: float = 0.0


class IngestionPipeline:
    def __init__(self, store, corpus, manifest: Manifest, embedder, gateway,
                 on_event: Callable[[str], None] | None = None) -> None:
        self.store = store
        self.corpus = corpus
        self.manifest = manifest
        self.embedder = embedder
        self.gateway = gateway
        self.chunker = HybridChunker(embedder)
        self.on_event = on_event

    def _emit(self, message: str) -> None:
        log.info(message)
        if self.on_event:
            self.on_event(message)

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    def scan(self) -> tuple[dict[str, Path], list[tuple[str, str]]]:
        files: dict[str, Path] = {}
        skipped: list[tuple[str, str]] = []
        root = settings.data_dir
        root.mkdir(parents=True, exist_ok=True)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            rel = path.relative_to(root).as_posix()
            kind, reason = classify(path)
            if kind in ("document", "image"):
                files[rel] = path
            else:
                skipped.append((rel, reason))
        return files, skipped

    def plan(self, force: bool = False) -> IngestPlan:
        files, skipped = self.scan()
        plan = IngestPlan(skipped=skipped)

        current: dict[str, FileInfo] = {}
        touched = False
        for rel, path in files.items():
            stat = path.stat()
            previous = self.manifest.files.get(rel)
            if previous and previous.get("size") == stat.st_size and previous.get("mtime_ns") == stat.st_mtime_ns:
                digest = previous["sha256"]  # cheap path: unchanged stat -> reuse the stored hash
            else:
                digest = sha256_file(path)
            current[rel] = FileInfo(rel, path, digest, stat.st_size, stat.st_mtime_ns)

        fingerprint = config_fingerprint()
        vector_count = self.store.count()
        recently_written = (time.time() - self.manifest.updated_at) < EMPTY_GRACE_SECONDS

        if force:
            plan.clear_first, plan.reason = True, "forced re-index"
        elif self.manifest.files and self.manifest.fingerprint != fingerprint:
            plan.clear_first, plan.reason = True, "ingestion settings changed"
        elif self.manifest.files and vector_count == 0 and not recently_written:
            plan.clear_first, plan.reason = True, "Pinecone namespace is empty"
        elif not self.manifest.files and vector_count > 0:
            plan.clear_first, plan.reason = True, "manifest missing, rebuilding cleanly"
        elif not self.manifest.files and current:
            plan.reason = "first run"

        if plan.clear_first:
            plan.to_add = dict(current)
            return plan

        for rel, info in current.items():
            previous = self.manifest.files.get(rel)
            if previous is None:
                plan.to_add[rel] = info
            elif previous.get("sha256") != info.sha256:
                plan.to_update[rel] = info
            else:
                plan.unchanged += 1
                if previous.get("size") != info.size or previous.get("mtime_ns") != info.mtime_ns:
                    previous["size"], previous["mtime_ns"] = info.size, info.mtime_ns
                    touched = True
        plan.to_delete = [rel for rel in self.manifest.files if rel not in current]

        if plan.needs_ingestion and plan.reason == "up to date":
            plan.reason = "data changed"
        if touched and not plan.needs_ingestion:
            self.manifest.save()
        return plan

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def run(self, plan: IngestPlan) -> IngestReport:
        started = time.time()
        report = IngestReport(reason=plan.reason, skipped=plan.skipped)

        if plan.clear_first:
            self._emit("Clearing existing vectors ...")
            self.store.delete_all()
            self.corpus.clear()
            self.manifest.reset()

        for rel in plan.to_delete:
            ids = self.manifest.files.get(rel, {}).get("ids", [])
            self._emit(f"Removing deleted file: {rel}")
            if ids:
                self.store.delete(ids)
                self.corpus.remove(ids)
            self.manifest.files.pop(rel, None)
            report.deleted.append(rel)

        todo = [(rel, info, True) for rel, info in plan.to_update.items()]
        todo += [(rel, info, False) for rel, info in plan.to_add.items()]
        for rel, info, is_update in todo:
            self._emit(f"{'Updating' if is_update else 'Ingesting'}: {rel}")
            old_ids = list(self.manifest.files.get(rel, {}).get("ids", []))
            try:
                count = self._ingest_file(info, old_ids)
            except ParseError as exc:
                self.manifest.files[rel] = {
                    "sha256": info.sha256, "size": info.size, "mtime_ns": info.mtime_ns,
                    "ids": [], "error": str(exc), "ingested_at": now_iso(),
                }
                if old_ids:
                    self.store.delete(old_ids)
                    self.corpus.remove(old_ids)
                report.failed.append((rel, str(exc)))
                self._emit(f"  skipped ({exc})")
                continue
            except Exception as exc:  # noqa: BLE001 - transient (network/API): retried on the next run
                log.exception("Ingestion failed for %s", rel)
                report.failed.append((rel, f"{type(exc).__name__}: {exc}"))
                continue
            report.vectors += count
            (report.updated if is_update else report.added).append(rel)

        self.manifest.set_fingerprint(config_fingerprint())
        self.manifest.save()
        self.corpus.save()
        report.seconds = time.time() - started
        return report

    # ------------------------------------------------------------------ #
    def _ingest_file(self, info: FileInfo, old_ids: list[str]) -> int:
        kind, _ = classify(info.path)
        records = self._image_records(info) if kind == "image" else self._document_records(info)
        if not records:
            raise ParseError("no extractable content")

        self.store.upsert(records)
        new_ids = [record["id"] for record in records]
        stale = [vector_id for vector_id in old_ids if vector_id not in set(new_ids)]
        if stale:
            self.store.delete(stale)
            self.corpus.remove(stale)
        self.corpus.add([to_corpus_item(r["id"], r["metadata"]) for r in records])

        self.manifest.files[info.rel] = {
            "sha256": info.sha256,
            "size": info.size,
            "mtime_ns": info.mtime_ns,
            "ids": new_ids,
            "modality": "image" if kind == "image" else "text",
            "chunks": len(new_ids),
            "ingested_at": now_iso(),
        }
        return len(new_ids)

    def _base_meta(self, info: FileInfo, vector_values, chunk_index: int) -> tuple[str, dict[str, Any]]:
        emb_hash = embedding_hash(vector_values)
        meta = {
            "source": info.rel,
            "file_name": info.path.name,
            "file_type": info.path.suffix.lower().lstrip("."),
            "chunk_index": chunk_index,
            "content_hash": info.sha256,
            "embedding_hash": emb_hash,
            "embedding_model": settings.embedding_model,
            "ingested_at": now_iso(),
        }
        return make_id(info.rel, chunk_index, emb_hash), meta

    # ---- text documents -------------------------------------------------- #
    def _document_records(self, info: FileInfo) -> list[dict[str, Any]]:
        parsed = parse_document(info.path, info.rel)
        pieces = [(section, text) for section in parsed.sections for text in self.chunker.split(section.text)]
        if not pieces:
            return []
        vectors = self.embedder.embed_documents([text for _, text in pieces])

        doc_meta = {f"doc_{k}": v for k, v in parsed.metadata.items() if v is not None}
        records = []
        for index, ((section, text), vector) in enumerate(zip(pieces, vectors)):
            vector_id, meta = self._base_meta(info, vector, index)
            meta.update(doc_meta)
            meta.update(
                {
                    "modality": "text",
                    "text": text,  # the chunk that was embedded
                    "raw_section": section.text,  # original extracted text of the page / slide / sheet
                    "location": section.location,
                    "page": section.page,
                    "ocr_used": section.ocr_used,
                    "total_chunks": len(pieces),
                }
            )
            records.append({"id": vector_id, "values": vector.tolist(), "metadata": fit_metadata(meta)})
        return records

    # ---- images ---------------------------------------------------------- #
    def _summarize_image(self, image: ParsedImage, file_name: str) -> dict[str, Any]:
        user = (
            f"File name: {file_name}\n"
            f"Image size: {image.width}x{image.height}\n"
            f"Tesseract OCR text (may contain errors):\n{truncate(image.ocr_text, 3000) or '(none)'}"
        )
        try:
            data = self.gateway.complete_json(
                "vision", IMAGE_SUMMARY_SYSTEM, user, images=[image.llm_data_url], max_tokens=900
            )
            if isinstance(data, dict) and data.get("summary"):
                return data
        except Exception as exc:  # noqa: BLE001
            log.warning("Vision summary failed for %s: %s", file_name, exc)
        fallback = truncate(image.ocr_text, 400) or "No description available."
        return {"summary": f"Image '{file_name}'. {fallback}", "description": "", "tags": [], "visible_text": image.ocr_text}

    def _image_records(self, info: FileInfo) -> list[dict[str, Any]]:
        image = parse_image(info.path, info.rel)
        analysis = self._summarize_image(image, info.path.name)

        tags = [str(t) for t in analysis.get("tags", []) if t][:20]
        visible = str(analysis.get("visible_text") or image.ocr_text or "")
        summary = str(analysis.get("summary", ""))
        description = str(analysis.get("description", ""))
        searchable = truncate(
            "\n".join(
                part
                for part in (
                    f"Image: {info.path.name}",
                    summary,
                    description,
                    f"Tags: {', '.join(tags)}" if tags else "",
                    f"Text in image: {visible}" if visible else "",
                )
                if part
            ),
            4000,
        )
        vector = self.embedder.embed_documents([searchable])[0]
        vector_id, meta = self._base_meta(info, vector, 0)
        meta.update(
            {
                "modality": "image",
                "text": searchable,  # what was embedded (summary + description + tags + OCR)
                "summary": summary,
                "description": description,
                "tags": tags,
                "ocr_text": image.ocr_text,
                "raw_section": visible,
                "location": "image",
                "image_b64": image.thumb_b64,  # JPEG thumbnail, base64 (fits Pinecone's 40 KB limit)
                "image_mime": "image/jpeg",
                "image_width": image.width,
                "image_height": image.height,
                "image_bytes": image.file_size,
                **{f"img_{k}": v for k, v in image.metadata.items()},
                "total_chunks": 1,
            }
        )
        return [{"id": vector_id, "values": vector.tolist(), "metadata": fit_metadata(meta)}]
