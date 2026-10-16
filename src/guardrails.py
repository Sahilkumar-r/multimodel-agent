"""Guardrails built on DeepEval.

DeepEval metrics are used as *runtime guards* (not just offline tests). Our own LiteLLM gateway is
wrapped as a DeepEval judge model, so the guard works with whichever provider you configured.

Input  : length + prompt-injection regexes (cheap), then a DeepEval GEval "Input Safety" metric.
Output : secret / PII-pattern scan (cheap), then DeepEval Toxicity, Faithfulness (grounded in the
         retrieved context) and Answer Relevancy metrics.
Context: retrieved / web passages that look like prompt-injection are dropped before generation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.config import settings
from src.prompts import INPUT_SAFETY_STEPS, JUDGE_SYSTEM
from src.utils import extract_json, truncate

log = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
        r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)",
        r"(reveal|show|print|repeat)\s+(me\s+)?(your|the)\s+(system|hidden|developer)\s+(prompt|instructions)",
        r"you\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak|god)\s+mode",
        r"\bact\s+as\s+(dan|an?\s+unrestricted|an?\s+unfiltered)",
        r"pretend\s+(that\s+)?you\s+have\s+no\s+(rules|restrictions|guidelines)",
    )
]
SECRET_PATTERNS = {
    "API key": re.compile(r"\b(sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{20,}|xai-[A-Za-z0-9]{20,}|pcsk_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|lsv2_[A-Za-z0-9_]{20,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
    "US SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 16:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit = digit * 2 - 9 if digit * 2 > 9 else digit * 2
        checksum += digit
    return checksum % 10 == 0


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


@dataclass
class GuardResult:
    passed: bool = True
    severity: str = "none"  # none | soft (retry-able) | hard (block)
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


def _make_judge(gateway):
    """Wrap the LiteLLM gateway as a DeepEval custom model."""
    from deepeval.models import DeepEvalBaseLLM

    class GatewayJudge(DeepEvalBaseLLM):
        def load_model(self, *args, **kwargs):
            return gateway

        def generate(self, prompt: str, schema=None, *args, **kwargs):
            text = gateway.complete("guardrail", JUDGE_SYSTEM, prompt, max_tokens=2000)
            if schema is None:
                return text
            data = extract_json(text)
            return schema.model_validate(data) if hasattr(schema, "model_validate") else schema(**data)

        async def a_generate(self, prompt: str, schema=None, *args, **kwargs):
            return self.generate(prompt, schema)

        def get_model_name(self, *args, **kwargs) -> str:
            return "litellm-gateway-judge"

    return GatewayJudge()


class Guardrails:
    def __init__(self, gateway) -> None:
        self.gateway = gateway
        self.enabled = settings.guardrails_enabled
        self._judge = None

    @property
    def judge(self):
        if self._judge is None:
            self._judge = _make_judge(self.gateway)
        return self._judge

    # ------------------------------------------------------------------ #
    def check_input(self, text: str) -> GuardResult:
        result = GuardResult()
        if not self.enabled:
            return result
        if len(text) > settings.max_input_chars:
            return GuardResult(False, "hard", [f"input longer than {settings.max_input_chars} characters"])
        if looks_like_injection(text):
            return GuardResult(False, "hard", ["possible prompt-injection / jailbreak attempt"])

        if settings.guard_input_llm:
            try:
                from deepeval.metrics import GEval
                from deepeval.test_case import LLMTestCase, LLMTestCaseParams

                metric = GEval(
                    name="Input Safety",
                    evaluation_steps=INPUT_SAFETY_STEPS,
                    evaluation_params=[LLMTestCaseParams.INPUT],
                    model=self.judge,
                    threshold=settings.guard_input_threshold,
                    async_mode=False,
                    verbose_mode=False,
                )
                metric.measure(LLMTestCase(input=text, actual_output="n/a"))
                result.scores["input_safety"] = float(metric.score or 0.0)
                if not metric.is_successful():
                    return GuardResult(
                        False, "hard", [f"unsafe request ({truncate(metric.reason or 'flagged by input guard', 160)})"],
                        result.scores,
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("Input guard (DeepEval) errored: %s", exc)
                if not settings.guard_fail_open:
                    return GuardResult(False, "hard", ["input guard unavailable"])
        return result

    def filter_context(self, texts: list[str]) -> list[bool]:
        """True for passages that are safe to show the generator (indirect-injection filter)."""
        if not self.enabled:
            return [True] * len(texts)
        return [not looks_like_injection(t) for t in texts]

    # ------------------------------------------------------------------ #
    def check_output(self, question: str, answer: str, contexts: list[str]) -> GuardResult:
        result = GuardResult()
        if not self.enabled or not answer.strip():
            return result

        if settings.guard_block_secrets:
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(answer):
                    return GuardResult(False, "hard", [f"answer contains a possible {label}"])
            for match in CARD_RE.finditer(answer):
                if _luhn_ok(match.group()):
                    return GuardResult(False, "hard", ["answer contains a possible payment-card number"])

        if not settings.guard_output_llm:
            return result

        try:
            from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ToxicityMetric
            from deepeval.test_case import LLMTestCase

            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=[truncate(c, 1500) for c in contexts[:8]] or None,
            )

            toxicity = ToxicityMetric(
                threshold=settings.guard_toxicity_threshold, model=self.judge, async_mode=False
            )
            toxicity.measure(test_case)
            result.scores["toxicity"] = float(toxicity.score or 0.0)
            if not toxicity.is_successful():
                return GuardResult(False, "hard", ["answer flagged as toxic"], result.scores)

            soft: list[str] = []
            relevancy = AnswerRelevancyMetric(
                threshold=settings.guard_relevancy_threshold, model=self.judge,
                include_reason=True, async_mode=False,
            )
            relevancy.measure(test_case)
            result.scores["answer_relevancy"] = float(relevancy.score or 0.0)
            if not relevancy.is_successful():
                soft.append(f"answer does not address the question well ({truncate(relevancy.reason or '', 200)})")

            if settings.guard_faithfulness and contexts:
                faithfulness = FaithfulnessMetric(
                    threshold=settings.guard_faithfulness_threshold, model=self.judge,
                    include_reason=True, async_mode=False,
                )
                faithfulness.measure(test_case)
                result.scores["faithfulness"] = float(faithfulness.score or 0.0)
                if not faithfulness.is_successful():
                    soft.append(f"claims not supported by the context ({truncate(faithfulness.reason or '', 300)})")

            if soft:
                return GuardResult(False, "soft", soft, result.scores)
        except Exception as exc:  # noqa: BLE001
            log.warning("Output guard (DeepEval) errored: %s", exc)
            if not settings.guard_fail_open:
                return GuardResult(False, "hard", ["output guard unavailable"], result.scores)
        return result
