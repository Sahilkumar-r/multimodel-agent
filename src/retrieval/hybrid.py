"""Hybrid search: dense (Pinecone) + sparse (BM25), fused with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import settings
from src.observability import traceable
from src.utils import rrf_fuse


@dataclass
class Candidate:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    rrf_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None

    def to_doc(self) -> dict[str, Any]:
        """Light, JSON-serialisable view stored in the LangGraph state (no image payloads)."""
        meta = self.metadata
        return {
            "id": self.id,
            "text": self.text,
            "source": meta.get("source", "unknown"),
            "file_type": meta.get("file_type", ""),
            "modality": meta.get("modality", "text"),
            "location": meta.get("location", ""),
            "page": meta.get("page"),
            "summary": meta.get("summary", ""),
            "rrf_score": round(self.rrf_score, 5),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rerank_score": None if self.rerank_score is None else round(float(self.rerank_score), 4),
        }


class HybridRetriever:
    def __init__(self, store, corpus, embedder) -> None:
        self.store = store
        self.corpus = corpus
        self.embedder = embedder

    @traceable(run_type="retriever", name="hybrid_search_rrf")
    def search(self, queries: list[str], pool_size: int | None = None) -> list[Candidate]:
        """Run dense + BM25 for every query variant and fuse all rankings with RRF."""
        pool_size = pool_size or settings.rrf_pool
        rankings: list[list[str]] = []
        candidates: dict[str, Candidate] = {}

        for query in queries:
            dense_hits = self.store.query(self.embedder.embed_query(query), settings.dense_top_k)
            sparse_hits = self.corpus.search(query, settings.sparse_top_k)
            rankings.append([hit["id"] for hit in dense_hits])
            rankings.append([vector_id for vector_id, _ in sparse_hits])

            for rank, hit in enumerate(dense_hits, start=1):
                cand = candidates.get(hit["id"])
                if cand is None:
                    cand = Candidate(id=hit["id"], text=hit["metadata"].get("text", ""), metadata=hit["metadata"])
                    candidates[hit["id"]] = cand
                else:  # dense hits carry the full metadata, so prefer them
                    cand.metadata = hit["metadata"]
                    cand.text = hit["metadata"].get("text", cand.text)
                cand.dense_rank = rank if cand.dense_rank is None else min(cand.dense_rank, rank)

            for rank, (vector_id, _score) in enumerate(sparse_hits, start=1):
                cand = candidates.get(vector_id)
                if cand is None:
                    item = self.corpus.items[vector_id]
                    cand = Candidate(id=vector_id, text=item["text"], metadata=dict(item["metadata"]))
                    candidates[vector_id] = cand
                cand.sparse_rank = rank if cand.sparse_rank is None else min(cand.sparse_rank, rank)

        fused = rrf_fuse(rankings, k=settings.rrf_k)
        for vector_id, score in fused.items():
            candidates[vector_id].rrf_score = score
        ordered = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)
        return ordered[:pool_size]
