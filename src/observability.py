"""LangSmith tracing helpers.

LangGraph runs are traced automatically once ``LANGSMITH_TRACING=true`` and an API key are set.
We additionally decorate the LLM gateway, hybrid search and the DuckDuckGo tool with
``@traceable`` so every model call, retrieval and tool call shows up as its own span.
"""
from __future__ import annotations

import logging
import os

from src import config  # noqa: F401  (ensures .env is loaded before we read the environment)

log = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    from langsmith import traceable
except ImportError:  # LangSmith is optional at runtime: fall back to a no-op decorator.

    def traceable(*dargs, **dkwargs):  # type: ignore[no-redef]
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        def decorator(func):
            return func

        return decorator


def setup_tracing() -> bool:
    """Configure LangSmith from the environment. Returns True when tracing is active."""
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    wanted = os.getenv("LANGSMITH_TRACING", "true").strip().lower() in {"1", "true", "yes", "on"}
    enabled = bool(api_key) and wanted

    if enabled:
        project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "multimodal-rag-agent"
        os.environ["LANGSMITH_API_KEY"] = api_key  # type: ignore[assignment]
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = project
        # Legacy names, still read by older langchain / langsmith releases.
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project
        log.info("LangSmith tracing enabled (project=%s)", project)
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if wanted and not api_key:
            log.warning("LANGSMITH_API_KEY not set: tracing disabled.")
    return enabled
