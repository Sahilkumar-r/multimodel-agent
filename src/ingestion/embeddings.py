"""Semantic embeddings (sentence-transformers). Vectors are L2-normalised, so dot product == cosine."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from src.config import settings

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embedding_model
        log.info("Loading embedding model %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([settings.query_instruction + text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
