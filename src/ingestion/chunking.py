"""Hybrid chunking: semantic boundaries first, recursive character splitting as a size guard.

1. Split the section into sentences and embed them.
2. Start a new chunk where the cosine distance between neighbouring sentences spikes
   (above a percentile of all distances) or when the size cap would be exceeded.
3. Any group that is still larger than ``chunk_size`` (e.g. one huge sentence or table) is split
   by LangChain's ``RecursiveCharacterTextSplitter`` (with overlap).
4. Tiny neighbouring chunks are merged back together.
"""
from __future__ import annotations

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.utils import normalize_whitespace, split_sentences


class HybridChunker:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.chunk_size = settings.chunk_size
        self.min_chars = settings.min_chunk_chars
        self.percentile = settings.semantic_percentile
        self.recursive = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        text = normalize_whitespace(text)
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        pieces: list[str] = []
        for group in self._semantic_groups(text):
            if len(group) <= self.chunk_size:
                pieces.append(group)
            else:
                pieces.extend(self.recursive.split_text(group))
        return self._merge_small([p.strip() for p in pieces if p.strip()])

    # ------------------------------------------------------------------ #
    def _semantic_groups(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if len(sentences) < 4:
            return [text]

        vectors = self.embedder.embed_documents(sentences)
        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.percentile(distances, self.percentile))

        groups: list[str] = []
        current: list[str] = [sentences[0]]
        current_len = len(sentences[0])
        for index, distance in enumerate(distances):
            nxt = sentences[index + 1]
            too_big = current_len + len(nxt) + 1 > self.chunk_size
            topic_shift = distance >= threshold and distance > 0 and current_len >= self.min_chars
            if too_big or topic_shift:
                groups.append(" ".join(current))
                current, current_len = [], 0
            current.append(nxt)
            current_len += len(nxt) + 1
        if current:
            groups.append(" ".join(current))
        return groups

    def _merge_small(self, pieces: list[str]) -> list[str]:
        merged: list[str] = []
        for piece in pieces:
            if merged and len(merged[-1]) < self.min_chars and len(merged[-1]) + len(piece) + 1 <= self.chunk_size:
                merged[-1] = f"{merged[-1]}\n{piece}"
            else:
                merged.append(piece)
        return merged
