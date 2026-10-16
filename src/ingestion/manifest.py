"""Manifest of what has been ingested: file hashes, vector ids and the config fingerprint.

The manifest is what makes ingestion *conditional*: on startup the pipeline hashes the files in
``data/`` and compares them with the manifest. Nothing is parsed, embedded or upserted unless a
file is new, changed or deleted, the ingestion config changed, or the Pinecone namespace is empty.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from src.config import settings
from src.utils import sha256_text


def config_fingerprint() -> str:
    """Hash of every setting that changes what gets stored (changing one forces a full re-ingest)."""
    payload = json.dumps(
        {
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "min_chunk_chars": settings.min_chunk_chars,
            "semantic_percentile": settings.semantic_percentile,
            "ocr_mode": settings.ocr_mode,
            "ocr_lang": settings.ocr_lang,
            "ocr_dpi": settings.ocr_dpi,
            "max_image_b64_bytes": settings.max_image_b64_bytes,
            "schema": 1,
        },
        sort_keys=True,
    )
    return sha256_text(payload)


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"version": 1, "config_fingerprint": None, "updated_at": 0.0, "files": {}}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("files"), dict):
                    self.data.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass  # corrupt manifest -> behave like a first run

    @property
    def files(self) -> dict[str, dict[str, Any]]:
        return self.data["files"]

    @property
    def fingerprint(self) -> str | None:
        return self.data.get("config_fingerprint")

    @property
    def updated_at(self) -> float:
        return float(self.data.get("updated_at") or 0.0)

    def set_fingerprint(self, value: str) -> None:
        self.data["config_fingerprint"] = value

    def reset(self) -> None:
        self.data["files"] = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
