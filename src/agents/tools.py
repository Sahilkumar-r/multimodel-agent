"""Tool layer. DuckDuckGo web search is the ONLY tool the agents may call (enforced by an allow-list)."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from src.observability import traceable

log = logging.getLogger(__name__)


@traceable(run_type="tool", name="duckduckgo_search")
def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS
    except ImportError:  # legacy package name
        from duckduckgo_search import DDGS  # type: ignore[no-redef]

    try:
        with DDGS() as client:
            raw = list(client.text(query, max_results=max_results, safesearch="moderate"))
    except Exception as exc:  # noqa: BLE001 - rate limits / network errors must not crash the agent
        log.warning("DuckDuckGo search failed for %r: %s", query, exc)
        return []

    results = []
    for item in raw:
        url = item.get("href") or item.get("url") or ""
        if not url:
            continue
        results.append(
            {
                "title": (item.get("title") or url).strip(),
                "url": url,
                "snippet": (item.get("body") or item.get("snippet") or "").strip(),
            }
        )
    return results


@tool
def duckduckgo_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the public web with DuckDuckGo. Returns a list of {title, url, snippet}."""
    return _ddg_search(query, max_results)


ALLOWED_TOOLS = {duckduckgo_search.name: duckduckgo_search}


def call_tool(name: str, **kwargs: Any) -> Any:
    """Single choke-point for tool use: anything outside the allow-list is refused."""
    if name not in ALLOWED_TOOLS:
        raise PermissionError(f"Tool '{name}' is not allowed. Allowed tools: {sorted(ALLOWED_TOOLS)}")
    return ALLOWED_TOOLS[name].invoke(kwargs)
