"""LLM gateway built on LiteLLM.

One ``Router`` exposes OpenAI, Anthropic and xAI (Grok) behind a single interface:
* aliases are ``<provider>-smart`` and ``<provider>-fast``
* each agent *role* maps to a tier (see ``ROLE_TIER``) and to the primary provider
* if the primary provider errors out, LiteLLM falls back to the other configured providers
"""
from __future__ import annotations

import logging
import os
from typing import Any

import litellm
from litellm import Router

from src.config import settings
from src.observability import traceable
from src.utils import extract_json

log = logging.getLogger(__name__)

litellm.drop_params = True  # silently drop params a given provider/model does not support
litellm.suppress_debug_info = True

# Which model tier each agent role uses. Override per role with MODEL_<ROLE>=<alias>.
ROLE_TIER = {
    "supervisor": "fast",
    "query_rewriter": "fast",
    "grader": "fast",
    "web_agent": "fast",
    "guardrail": "fast",
    "answer": "smart",
    "vision": "smart",
}


def _redact_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep base64 images out of LangSmith traces."""
    redacted = dict(inputs)
    images = redacted.get("images")
    if images:
        redacted["images"] = f"<{len(images)} image(s) omitted>"
    return redacted


class LLMGateway:
    def __init__(self) -> None:
        self.providers = settings.available_providers()
        if not self.providers:
            raise RuntimeError(
                "No LLM API key found. Set at least one of OPENAI_API_KEY, ANTHROPIC_API_KEY "
                "or XAI_API_KEY in your .env file."
            )
        primary = settings.primary_provider
        if primary not in self.providers:
            log.warning("PRIMARY_PROVIDER=%s has no API key; using %s", primary, self.providers[0])
            primary = self.providers[0]
        self.primary = primary
        self._order = [primary] + [p for p in self.providers if p != primary]
        self.router = Router(
            model_list=self._model_list(),
            fallbacks=self._fallbacks(),
            num_retries=settings.llm_retries,
            timeout=settings.llm_timeout,
        )
        self.aliases = {entry["model_name"] for entry in self._model_list()}

    # ------------------------------------------------------------------ #
    def _model_list(self) -> list[dict[str, Any]]:
        entries = []
        for provider in self.providers:
            for tier in ("smart", "fast"):
                entries.append(
                    {
                        "model_name": f"{provider}-{tier}",
                        "litellm_params": {
                            "model": f"{provider}/{settings.model_name(provider, tier)}",
                            "api_key": settings.api_key(provider),
                        },
                    }
                )
        return entries

    def _fallbacks(self) -> list[dict[str, list[str]]]:
        fallbacks = []
        for provider in self._order:
            others = [p for p in self._order if p != provider]
            if not others:
                continue
            for tier in ("smart", "fast"):
                fallbacks.append({f"{provider}-{tier}": [f"{o}-{tier}" for o in others]})
        return fallbacks

    def model_for(self, role: str) -> str:
        override = os.getenv(f"MODEL_{role.upper()}", "").strip()
        if override:
            if override in self.aliases:
                return override
            log.warning("MODEL_%s=%s is not a configured alias %s", role.upper(), override, sorted(self.aliases))
        return f"{self.primary}-{ROLE_TIER.get(role, 'fast')}"

    # ------------------------------------------------------------------ #
    @traceable(run_type="llm", name="llm_gateway", process_inputs=_redact_inputs)
    def complete(
        self,
        role: str,
        system: str,
        user: str,
        *,
        history: list[dict[str, str]] | None = None,
        images: list[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> str:
        """Single chat completion. ``images`` are data-URLs (``data:image/jpeg;base64,...``)."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        if images:
            content: Any = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": url}} for url in images
            ]
        else:
            content = user
        messages.append({"role": "user", "content": content})

        response = self.router.completion(
            model=self.model_for(role),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def complete_json(self, role: str, system: str, user: str, **kwargs: Any) -> Any:
        """Like ``complete`` but parses a JSON reply (one automatic repair retry)."""
        text = self.complete(role, system, user, **kwargs)
        try:
            return extract_json(text)
        except ValueError:
            repair = self.complete(
                role,
                system,
                user + "\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON object.",
                **kwargs,
            )
            return extract_json(repair)
