"""Central configuration. Every value can be overridden from the .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PROVIDERS = ("openai", "anthropic", "xai")
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
}
# Defaults are only starting points: override with <PROVIDER>_SMART_MODEL / _FAST_MODEL in .env
# (check each provider's current model list, names change often).
DEFAULT_MODELS = {
    "openai": {"smart": "gpt-4o", "fast": "gpt-4o-mini"},
    "anthropic": {"smart": "claude-sonnet-5-5", "fast": "claude-haiku-5-5"},
    "xai": {"smart": "grok-4", "fast": "grok-3-mini"},
}


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    path = Path(_str(name, str(default))).expanduser()
    return path if path.is_absolute() else ROOT / path


class Settings:
    def __init__(self) -> None:
        # ---- paths -------------------------------------------------------- #
        self.data_dir = _path("DATA_DIR", ROOT / "data")
        self.cache_dir = _path("CACHE_DIR", ROOT / ".cache")
        self.log_level = _str("LOG_LEVEL", "WARNING").upper()

        # ---- Pinecone ----------------------------------------------------- #
        self.pinecone_api_key = _str("PINECONE_API_KEY")
        self.pinecone_index = _str("PINECONE_INDEX", "multimodal-rag-agent")
        self.pinecone_namespace = _str("PINECONE_NAMESPACE", "default")
        self.pinecone_cloud = _str("PINECONE_CLOUD", "aws")
        self.pinecone_region = _str("PINECONE_REGION", "us-east-1")

        # ---- embeddings / reranker --------------------------------------- #
        self.embedding_model = _str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.query_instruction = os.getenv(
            "QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: "
        )
        self.reranker_model = _str("RERANKER_MODEL", "BAAI/bge-base-en-v1.5")

        # ---- chunking ----------------------------------------------------- #
        self.chunk_size = _int("CHUNK_SIZE", 1000)
        self.chunk_overlap = _int("CHUNK_OVERLAP", 150)
        self.min_chunk_chars = _int("MIN_CHUNK_CHARS", 250)
        self.semantic_percentile = _float("SEMANTIC_BREAKPOINT_PERCENTILE", 85.0)

        # ---- OCR / images ------------------------------------------------- #
        self.ocr_mode = _str("OCR_MODE", "auto").lower()  # auto | always | off
        self.ocr_lang = _str("OCR_LANG", "eng")
        self.ocr_dpi = _int("OCR_DPI", 200)
        self.ocr_min_chars = _int("OCR_MIN_CHARS", 40)
        self.ocr_min_image_px = _int("OCR_MIN_IMAGE_PX", 100)
        self.tesseract_cmd = _str("TESSERACT_CMD")
        self.max_image_b64_bytes = _int("MAX_IMAGE_B64_BYTES", 28_000)
        self.vision_max_side = _int("VISION_MAX_SIDE", 1568)

        # ---- retrieval ---------------------------------------------------- #
        self.dense_top_k = _int("DENSE_TOP_K", 20)
        self.sparse_top_k = _int("SPARSE_TOP_K", 20)
        self.rrf_k = _int("RRF_K", 60)
        self.rrf_pool = _int("RRF_POOL", 30)
        self.rerank_top_n = _int("RERANK_TOP_N", 6)
        self.query_variants = _int("QUERY_VARIANTS", 2)

        # ---- corrective RAG ---------------------------------------------- #
        self.crag_min_relevant = _int("CRAG_MIN_RELEVANT", 2)
        self.rerank_fallback_threshold = _float("RERANK_FALLBACK_THRESHOLD", 0.5)
        self.grade_max_chars = _int("GRADE_MAX_CHARS", 700)
        self.web_max_results = _int("WEB_MAX_RESULTS", 5)

        # ---- agents ------------------------------------------------------- #
        self.primary_provider = _str("PRIMARY_PROVIDER", "openai").lower()
        self.max_supervisor_steps = _int("MAX_SUPERVISOR_STEPS", 4)
        self.max_answer_retries = _int("MAX_ANSWER_RETRIES", 1)
        self.history_messages = _int("HISTORY_MESSAGES", 6)
        self.llm_timeout = _int("LLM_TIMEOUT", 120)
        self.llm_retries = _int("LLM_RETRIES", 2)

        # ---- guardrails (DeepEval) --------------------------------------- #
        self.guardrails_enabled = _bool("GUARDRAILS_ENABLED", True)
        self.guard_input_llm = _bool("GUARD_INPUT_LLM", True)
        self.guard_output_llm = _bool("GUARD_OUTPUT_LLM", True)
        self.guard_faithfulness = _bool("GUARD_FAITHFULNESS", True)
        self.guard_fail_open = _bool("GUARD_FAIL_OPEN", True)
        self.guard_block_secrets = _bool("GUARD_BLOCK_SECRETS", True)
        self.guard_input_threshold = _float("GUARD_INPUT_THRESHOLD", 0.5)
        self.guard_faithfulness_threshold = _float("GUARD_FAITHFULNESS_THRESHOLD", 0.7)
        self.guard_relevancy_threshold = _float("GUARD_RELEVANCY_THRESHOLD", 0.6)
        self.guard_toxicity_threshold = _float("GUARD_TOXICITY_THRESHOLD", 0.5)
        self.max_input_chars = _int("MAX_INPUT_CHARS", 6000)

    # ---- providers -------------------------------------------------------- #
    def api_key(self, provider: str) -> str:
        return _str(API_KEY_ENV[provider])

    def available_providers(self) -> list[str]:
        return [p for p in PROVIDERS if self.api_key(p)]

    def model_name(self, provider: str, tier: str) -> str:
        env = f"{provider.upper()}_{tier.upper()}_MODEL"
        return _str(env, DEFAULT_MODELS[provider][tier])


settings = Settings()
