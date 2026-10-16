"""Application facade: builds every component once and exposes ingest / upload / ask."""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage

from src.agents.graph import build_graph
from src.config import settings
from src.guardrails import Guardrails
from src.ingestion.embeddings import get_embedder
from src.ingestion.manifest import Manifest
from src.ingestion.parsers import classify
from src.ingestion.pipeline import IngestionPipeline, IngestReport
from src.llm_gateway import LLMGateway
from src.observability import setup_tracing
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import BiEncoderReranker
from src.retrieval.sparse import LocalCorpus
from src.retrieval.vector_store import PineconeStore
from src.utils import sha256_file

URL_RE = re.compile(r"^(https?://|ftp://|www\.)", re.I)


@dataclass
class AnswerResult:
    answer: str
    route: list[str] = field(default_factory=list)
    verdict: str = ""


class RagApp:
    def __init__(self, on_event: Callable[[str], None] | None = None) -> None:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.tracing = setup_tracing()

        self.gateway = LLMGateway()
        self.embedder = get_embedder()
        self.store = PineconeStore(self.embedder.dim)
        self.manifest = Manifest(settings.cache_dir / "manifest.json")
        self.corpus = LocalCorpus(settings.cache_dir / "chunks.jsonl")
        self.reranker = BiEncoderReranker()
        self.retriever = HybridRetriever(self.store, self.corpus, self.embedder)
        self.guardrails = Guardrails(self.gateway)
        self.pipeline = IngestionPipeline(
            self.store, self.corpus, self.manifest, self.embedder, self.gateway, on_event=on_event
        )
        self.graph = build_graph(
            self.gateway, self.guardrails, self.store, self.retriever, self.reranker, self.kb_overview
        )

    # ------------------------------------------------------------------ #
    def kb_overview(self) -> str:
        files = [name for name, meta in self.manifest.files.items() if not meta.get("error")]
        if not files:
            return "empty (no documents or images ingested yet)"
        shown = ", ".join(files[:15]) + (f" (+{len(files) - 15} more)" if len(files) > 15 else "")
        return f"{len(files)} file(s): {shown}"

    def sources(self) -> list[tuple[str, str, int, str]]:
        """(file, modality, vectors, note) for every tracked file."""
        return [
            (name, meta.get("modality", "?"), len(meta.get("ids", [])), meta.get("error", ""))
            for name, meta in sorted(self.manifest.files.items())
        ]

    def _load_corpus(self) -> None:
        if len(self.corpus) == 0 and not self.corpus.load() and self.manifest.files and self.store.count() > 0:
            self.corpus.rebuild_from_store(self.store)

    # ------------------------------------------------------------------ #
    def ensure_ingested(self, force: bool = False) -> tuple[IngestReport | None, str]:
        """Ingest only when needed. Returns (report or None if skipped, reason)."""
        plan = self.pipeline.plan(force=force)
        if not plan.clear_first:
            self._load_corpus()  # must happen before run(): the run saves the merged corpus
        if not plan.needs_ingestion:
            return None, plan.reason
        return self.pipeline.run(plan), plan.reason

    def upload(self, raw_path: str) -> IngestReport | None:
        """Copy a document/image into data/uploads and ingest just what changed."""
        raw = raw_path.strip().strip("\"'")
        if URL_RE.match(raw):
            raise ValueError("URLs and website links are not supported. Upload a file instead.")
        source = Path(raw).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"File not found: {source}")
        kind, reason = classify(source)
        if kind in ("blocked", "unsupported"):
            raise ValueError(reason)

        target_dir = settings.data_dir / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        inside_data = settings.data_dir.resolve() in source.resolve().parents
        if not inside_data:
            target = target_dir / source.name
            if target.exists() and sha256_file(target) != sha256_file(source):
                target = target_dir / f"{source.stem}-{sha256_file(source)[:8]}{source.suffix}"
            if not target.exists():
                shutil.copy2(source, target)

        plan = self.pipeline.plan()
        if not plan.clear_first:
            self._load_corpus()
        return self.pipeline.run(plan) if plan.needs_ingestion else None

    # ------------------------------------------------------------------ #
    def ask(self, question: str, thread_id: str = "default") -> AnswerResult:
        state = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={
                "configurable": {"thread_id": thread_id},
                "run_name": "multimodal_rag_agent",
                "tags": ["rag", "supervisor"],
                "recursion_limit": 40,
            },
        )
        return AnswerResult(
            answer=state.get("final_answer") or state.get("answer", ""),
            route=state.get("route_log", []),
            verdict=state.get("crag_verdict", ""),
        )
