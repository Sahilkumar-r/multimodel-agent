"""Offline tests for the dependency-light helpers (no API keys or models needed).

Run with:  python -m pytest -q
"""
import json

import pytest

from src.utils import (
    clean_metadata,
    embedding_hash,
    extract_json,
    fit_metadata,
    make_id,
    normalize_whitespace,
    rrf_fuse,
    sha256_file,
    split_sentences,
    tokenize,
)


def test_rrf_prefers_documents_ranked_well_in_both_lists():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    scores = rrf_fuse([dense, sparse], k=60)
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert set(ranked[:2]) == {"a", "b"}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["d"] < scores["c"] or scores["d"] == pytest.approx(scores["c"])


def test_rrf_weights():
    scores = rrf_fuse([["a"], ["b"]], k=60, weights=[2.0, 1.0])
    assert scores["a"] > scores["b"]


def test_embedding_hash_is_stable_and_sensitive():
    vec = [0.1, -0.2, 0.3, 0.0]
    assert embedding_hash(vec) == embedding_hash(list(vec))
    assert embedding_hash(vec) == embedding_hash([0.1, -0.2, 0.3, -0.0])  # -0.0 normalised
    assert embedding_hash(vec) == embedding_hash([0.1000001, -0.2, 0.3, 0.0])  # jitter below 5 decimals
    assert embedding_hash(vec) != embedding_hash([0.1, -0.2, 0.31, 0.0])


def test_make_id_is_deterministic_and_unique_per_source():
    assert make_id("a.pdf", 0, "h") == make_id("a.pdf", 0, "h")
    assert make_id("a.pdf", 0, "h") != make_id("b.pdf", 0, "h")
    assert make_id("a.pdf", 0, "h") != make_id("a.pdf", 1, "h")


def test_sha256_file_changes_with_content(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("hello")
    first = sha256_file(path)
    path.write_text("hello!")
    assert first != sha256_file(path)


def test_fit_metadata_truncates_big_text_but_keeps_image():
    meta = {"text": "x" * 60_000, "image_b64": "A" * 20_000, "source": "a.png", "none": None}
    fitted = fit_metadata(meta)
    assert len(json.dumps(fitted, ensure_ascii=False).encode()) <= 38_000
    assert fitted["image_b64"] == "A" * 20_000
    assert "none" not in fitted


def test_clean_metadata_coerces_types():
    cleaned = clean_metadata({"a": 1, "b": ["x", None, "y"], "c": {"k": 1}, "d": True, "e": []})
    assert cleaned["a"] == 1 and cleaned["b"] == ["x", "y"] and cleaned["d"] is True
    assert isinstance(cleaned["c"], str)
    assert "e" not in cleaned


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"next": "answer_agent"} Hope it helps.') == {"next": "answer_agent"}
    assert extract_json("[1, 2, 3]") == [1, 2, 3]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_split_sentences_and_whitespace():
    parts = split_sentences("Hello world. This is a Test.\n\nNew paragraph here! Another one?")
    assert parts == ["Hello world.", "This is a Test.", "New paragraph here!", "Another one?"]
    assert normalize_whitespace("a  b\r\n\r\n\r\n\r\nc \x00") == "a b\n\nc"


def test_tokenize():
    assert tokenize("Hello, World_2! caf\u00e9") == ["hello", "world_2", "caf\u00e9"]
