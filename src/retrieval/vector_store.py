"""Pinecone wrapper: index creation, batched upsert, query, fetch, delete and enumeration."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterator

from src.config import settings

log = logging.getLogger(__name__)

MAX_BATCH_BYTES = 1_500_000  # Pinecone request limit is 2 MB
MAX_BATCH_VECTORS = 100


class PineconeStore:
    def __init__(self, dim: int) -> None:
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set. Add it to your .env file.")
        from pinecone import Pinecone, ServerlessSpec

        self._spec_cls = ServerlessSpec
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.namespace = settings.pinecone_namespace
        self.dim = dim
        self._ensure_index()
        self.index = self.pc.Index(settings.pinecone_index)

    # ------------------------------------------------------------------ #
    def _ensure_index(self) -> None:
        name = settings.pinecone_index
        if name not in self.pc.list_indexes().names():
            log.info("Creating Pinecone index %s (dim=%s)", name, self.dim)
            self.pc.create_index(
                name=name,
                dimension=self.dim,
                metric="cosine",
                spec=self._spec_cls(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        description = self.pc.describe_index(name)
        if int(description.dimension) != self.dim:
            raise RuntimeError(
                f"Pinecone index '{name}' has dimension {description.dimension} but the embedding model "
                f"'{settings.embedding_model}' produces {self.dim}. Use another PINECONE_INDEX or recreate it."
            )
        for _ in range(60):
            if self.pc.describe_index(name).status["ready"]:
                return
            time.sleep(2)
        raise RuntimeError(f"Pinecone index '{name}' did not become ready in time.")

    # ------------------------------------------------------------------ #
    def count(self) -> int:
        stats = self.index.describe_index_stats()
        data = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        namespaces = data.get("namespaces") or {}
        return int((namespaces.get(self.namespace) or {}).get("vector_count", 0))

    def upsert(self, records: list[dict[str, Any]]) -> None:
        batch: list[dict[str, Any]] = []
        batch_bytes = 0
        for record in records:
            size = len(json.dumps(record["metadata"], ensure_ascii=False)) + len(record["values"]) * 8
            if batch and (len(batch) >= MAX_BATCH_VECTORS or batch_bytes + size > MAX_BATCH_BYTES):
                self.index.upsert(vectors=batch, namespace=self.namespace)
                batch, batch_bytes = [], 0
            batch.append(record)
            batch_bytes += size
        if batch:
            self.index.upsert(vectors=batch, namespace=self.namespace)

    def query(self, vector, top_k: int) -> list[dict[str, Any]]:
        response = self.index.query(
            vector=[float(x) for x in vector],
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
        return [
            {"id": match.id, "score": float(match.score), "metadata": dict(match.metadata or {})}
            for match in response.matches
        ]

    def fetch(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return {id: metadata} for the given vector ids."""
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(ids), 100):
            response = self.index.fetch(ids=ids[start : start + 100], namespace=self.namespace)
            for vector_id, vector in response.vectors.items():
                out[vector_id] = dict(vector.metadata or {})
        return out

    def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), 500):
            self.index.delete(ids=ids[start : start + 500], namespace=self.namespace)

    def delete_all(self) -> None:
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
        except Exception as exc:  # noqa: BLE001 - "namespace not found" on an empty namespace
            log.debug("delete_all ignored: %s", exc)

    def iter_records(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Enumerate every (id, metadata) in the namespace (serverless indexes)."""
        for id_page in self.index.list(namespace=self.namespace):
            ids = list(id_page)
            for vector_id, metadata in self.fetch(ids).items():
                yield vector_id, metadata
