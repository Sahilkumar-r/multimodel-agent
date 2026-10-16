"""Sparse (BM25) side of hybrid search.

Chunk texts live in Pinecone metadata (source of truth). A JSONL cache next to the manifest keeps a
local copy so BM25 can be built without re-reading Pinecone on every start. If the cache is missing
but Pinecone has vectors, it is rebuilt from Pinecone.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import tokenize

log = logging.getLogger(__name__)

LITE_KEYS = (
    "source", "file_name", "file_type", "modality", "location", "page", "chunk_index", "summary",
)


def to_corpus_item(vector_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": vector_id,
        "text": metadata.get("text", ""),
        "metadata": {key: metadata[key] for key in LITE_KEYS if key in metadata},
    }


class LocalCorpus:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: dict[str, dict[str, Any]] = {}
        self._bm25 = None
        self._ids: list[str] = []
        self._dirty = True

    def __len__(self) -> int:
        return len(self.items)

    # ---- persistence ---------------------------------------------------- #
    def load(self) -> bool:
        if not self.path.exists():
            return False
        self.items = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.items[item["id"]] = item
        self._dirty = True
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for item in self.items.values():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    # ---- mutation ------------------------------------------------------- #
    def add(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            self.items[item["id"]] = item
        self._dirty = True

    def remove(self, ids: list[str]) -> None:
        for vector_id in ids:
            self.items.pop(vector_id, None)
        self._dirty = True

    def clear(self) -> None:
        self.items = {}
        self._dirty = True

    def rebuild_from_store(self, store) -> int:
        log.info("Rebuilding local BM25 corpus from Pinecone ...")
        self.items = {vid: to_corpus_item(vid, meta) for vid, meta in store.iter_records()}
        self._dirty = True
        self.save()
        return len(self.items)

    # ---- search --------------------------------------------------------- #
    def _build(self) -> None:
        from rank_bm25 import BM25Okapi

        self._ids = list(self.items.keys())
        tokenized = [tokenize(self.items[i]["text"]) or ["_"] for i in self._ids]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._dirty = False

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.items:
            return []
        if self._dirty or self._bm25 is None:
            self._build()
        tokens = tokenize(query)
        if not tokens or self._bm25 is None:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens))
        top = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in top if scores[i] > 0]
