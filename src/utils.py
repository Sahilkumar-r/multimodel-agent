"""Small, dependency-light helpers shared across the project."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_hash(vector: Sequence[float]) -> str:
    """Content hash of an embedding vector.

    Values are rounded to 5 decimals (and -0.0 normalised to 0.0) so that tiny
    floating-point jitter between runs does not change the hash.
    """
    import numpy as np

    array = np.round(np.asarray(vector, dtype=np.float32), 5) + np.float32(0.0)
    return hashlib.sha256(array.tobytes()).hexdigest()


def make_id(source: str, chunk_index: int, emb_hash: str) -> str:
    """Deterministic, content-addressed vector id (stable across runs)."""
    return hashlib.sha1(f"{source}|{chunk_index}|{emb_hash}".encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Pinecone metadata
# --------------------------------------------------------------------------- #

# Pinecone allows 40 KB of metadata per vector; keep a safety margin.
MAX_METADATA_BYTES = 38_000


def _json_size(meta: dict[str, Any]) -> int:
    return len(json.dumps(meta, ensure_ascii=False).encode("utf-8"))


def clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce a dict into what Pinecone accepts: str | int | float | bool | list[str]."""
    cleaned: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple, set)):
            items = [str(v) for v in value if v is not None]
            if items:
                cleaned[key] = items
        else:
            cleaned[key] = str(value)
    return cleaned


def fit_metadata(
    meta: dict[str, Any],
    limit: int = MAX_METADATA_BYTES,
    protected: tuple[str, ...] = ("image_b64",),
) -> dict[str, Any]:
    """Clean metadata and truncate the largest string fields until it fits ``limit`` bytes."""
    meta = clean_metadata(meta)
    for _ in range(60):
        if _json_size(meta) <= limit:
            return meta
        candidates = [
            (len(value), key)
            for key, value in meta.items()
            if isinstance(value, str) and key not in protected and len(value) > 64
        ]
        if not candidates:
            break
        _, key = max(candidates)
        meta[key] = meta[key][: int(len(meta[key]) * 0.7)]
    # Last resort: drop protected payloads rather than failing the upsert.
    for key in protected:
        meta.pop(key, None)
    return meta


# --------------------------------------------------------------------------- #
# Retrieval helpers
# --------------------------------------------------------------------------- #


def rrf_fuse(
    rankings: Iterable[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d))."""
    scores: dict[str, float] = {}
    for idx, ranking in enumerate(rankings):
        weight = weights[idx] if weights else 1.0
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n{2,}|\n(?=[-*\u2022]\s)")


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENT_SPLIT.split(text)]
    return [part for part in parts if part]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "\u2026"


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


# --------------------------------------------------------------------------- #
# LLM output parsing
# --------------------------------------------------------------------------- #


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM reply, tolerating code fences and surrounding prose."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON found in model output: {text[:200]!r}")
