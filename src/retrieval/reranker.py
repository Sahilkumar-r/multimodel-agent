"""Bi-encoder re-ranking.

A bi-encoder embeds the query and every candidate independently and scores them by cosine
similarity. We use a *different* (larger) model than the one used at ingestion time, so the
re-ranking is a genuine second opinion on the RRF-fused candidate pool.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from src.config import settings
from src.retrieval.hybrid import Candidate
from src.utils import split_sentences

log = logging.getLogger(__name__)


class BiEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = None  # lazy: loaded on first use

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading re-ranker bi-encoder %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Cosine similarity between the query and each text."""
        if not texts:
            return np.zeros(0, dtype=np.float32)
        query_vec = self.model.encode(
            [settings.query_instruction + query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        text_vecs = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(text_vecs @ query_vec, dtype=np.float32)

    def rerank(self, query: str, candidates: list[Candidate], top_n: int | None = None) -> list[Candidate]:
        top_n = top_n or settings.rerank_top_n
        if not candidates:
            return []
        scores = self.score(query, [c.text for c in candidates])
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_n]

    def refine(self, query: str, text: str, keep_ratio: float = 0.5, min_keep: int = 3) -> str:
        """Corrective-RAG 'knowledge refinement': keep only the sentences most related to the query."""
        sentences = split_sentences(text)
        if len(sentences) <= min_keep:
            return text
        scores = self.score(query, sentences)
        keep = max(min_keep, math.ceil(len(sentences) * keep_ratio))
        kept_idx = sorted(np.argsort(-scores)[:keep].tolist())  # restore original order
        return " ".join(sentences[i] for i in kept_idx)
