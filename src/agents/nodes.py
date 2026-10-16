"""Supervisor + sub-agent nodes for the LangGraph graph."""
from __future__ import annotations

import logging
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage

from src.agents.crag import format_history
from src.agents.state import AgentState
from src.agents.tools import call_tool
from src.config import settings
from src.prompts import ANSWER_SYSTEM, SUPERVISOR_SYSTEM, WEB_AGENT_SYSTEM
from src.utils import dedupe, truncate

log = logging.getLogger(__name__)

ROUTES = ("retrieval_agent", "web_search_agent", "answer_agent")
CAUTION = (
    "\n\n> \u26a0\ufe0f _Automated checks could not fully verify every claim in this answer against the "
    "retrieved sources. Please double-check anything important._"
)


class AgentNodes:
    def __init__(self, gateway, guardrails, store, kb_overview: Callable[[], str]) -> None:
        self.gateway = gateway
        self.guardrails = guardrails
        self.store = store
        self.kb_overview = kb_overview

    # ------------------------------------------------------------------ #
    # Guards
    # ------------------------------------------------------------------ #
    def input_guard(self, state: AgentState) -> dict[str, Any]:
        question = str(state["messages"][-1].content)
        reset: dict[str, Any] = {
            "question": question, "steps": 0, "route_log": ["input_guard"], "next_agent": "",
            "rewritten_queries": [], "candidates": [], "ranked": [], "relevant_docs": [],
            "crag_verdict": "", "retrieval_done": False, "web_results": [], "web_done": False,
            "contexts": [], "answer": "", "final_answer": "", "guard_feedback": "",
            "retry_answer": False, "retries": 0, "blocked": False,
        }
        verdict = self.guardrails.check_input(question)
        if verdict.passed:
            return reset
        reset.update(
            blocked=True,
            answer="I can't help with that request. It was blocked by my safety checks "
                   f"({'; '.join(verdict.reasons)}). Feel free to rephrase it.",
            route_log=["input_guard:blocked"],
        )
        return reset

    def route_after_input_guard(self, state: AgentState) -> str:
        return "finalize" if state.get("blocked") else "supervisor"

    def output_guard(self, state: AgentState) -> dict[str, Any]:
        verdict = self.guardrails.check_output(state["question"], state["answer"], state.get("contexts", []))
        log_entry = f"output_guard:{'pass' if verdict.passed else verdict.severity}"
        route_log = state.get("route_log", []) + [log_entry]
        if verdict.passed:
            return {"retry_answer": False, "route_log": route_log}
        if verdict.severity == "hard":
            return {
                "retry_answer": False,
                "answer": "I can't share that response because it failed my safety checks "
                          f"({'; '.join(verdict.reasons)}).",
                "route_log": route_log,
            }
        retries = state.get("retries", 0)
        if retries < settings.max_answer_retries:
            return {
                "retry_answer": True,
                "retries": retries + 1,
                "guard_feedback": "; ".join(verdict.reasons),
                "route_log": route_log,
            }
        return {"retry_answer": False, "answer": state["answer"] + CAUTION, "route_log": route_log}

    def route_after_output_guard(self, state: AgentState) -> str:
        return "answer_agent" if state.get("retry_answer") else "finalize"

    # ------------------------------------------------------------------ #
    # Supervisor
    # ------------------------------------------------------------------ #
    def supervisor(self, state: AgentState) -> dict[str, Any]:
        steps = state.get("steps", 0)
        retrieval_done, web_done = state.get("retrieval_done", False), state.get("web_done", False)
        route_log = state.get("route_log", [])

        if steps >= settings.max_supervisor_steps:
            return {"next_agent": "answer_agent", "steps": steps + 1, "route_log": route_log + ["supervisor->answer(budget)"]}

        progress = (
            f"retrieval_agent run: {'yes' if retrieval_done else 'no'}"
            + (f" (verdict={state.get('crag_verdict')}, relevant_chunks={len(state.get('relevant_docs', []))})" if retrieval_done else "")
            + f"\nweb_search_agent run: {'yes' if web_done else 'no'} (web_results={len(state.get('web_results', []))})"
            + f"\nsteps used: {steps}/{settings.max_supervisor_steps}"
        )
        user = (
            f"Conversation so far:\n{format_history(state.get('messages', [])[:-1], settings.history_messages)}\n\n"
            f"Latest user message: {state['question']}\n\n"
            f"Knowledge base: {self.kb_overview()}\n\nProgress:\n{progress}"
        )
        decision, reason = None, ""
        try:
            data = self.gateway.complete_json("supervisor", SUPERVISOR_SYSTEM, user, max_tokens=200)
            decision, reason = data.get("next"), str(data.get("reason", ""))
        except Exception as exc:  # noqa: BLE001
            log.warning("Supervisor failed, using default routing: %s", exc)

        if decision not in ROUTES:
            decision = "answer_agent" if retrieval_done else "retrieval_agent"
        if (decision == "retrieval_agent" and retrieval_done) or (decision == "web_search_agent" and web_done):
            decision = "answer_agent"  # never repeat a specialist
        return {
            "next_agent": decision,
            "supervisor_reason": reason,
            "steps": steps + 1,
            "route_log": route_log + [f"supervisor->{decision.replace('_agent', '')}"],
        }

    def route_from_supervisor(self, state: AgentState) -> str:
        return state.get("next_agent", "answer_agent")

    # ------------------------------------------------------------------ #
    # Sub-agents
    # ------------------------------------------------------------------ #
    def web_search_agent(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        queries = [question]
        try:
            data = self.gateway.complete_json(
                "web_agent",
                WEB_AGENT_SYSTEM,
                f"Conversation so far:\n{format_history(state.get('messages', [])[:-1], settings.history_messages)}\n\n"
                f"Question: {question}",
                max_tokens=200,
            )
            queries = dedupe([q for q in data.get("queries", []) if isinstance(q, str)])[:2] or [question]
        except Exception as exc:  # noqa: BLE001
            log.warning("Web query generation failed, searching the raw question: %s", exc)

        collected = {r["url"]: r for r in state.get("web_results", [])}
        for query in queries:
            for result in call_tool("duckduckgo_search", query=query, max_results=settings.web_max_results):
                collected.setdefault(result["url"], result)
        return {
            "web_results": list(collected.values())[:8],
            "web_done": True,
            "route_log": state.get("route_log", []) + [f"web_search({len(collected)})"],
        }

    def answer_agent(self, state: AgentState) -> dict[str, Any]:
        docs = state.get("relevant_docs", [])
        web = state.get("web_results", [])

        kb_safe = self.guardrails.filter_context([d["text"] for d in docs])
        docs = [d for d, ok in zip(docs, kb_safe) if ok]
        web_safe = self.guardrails.filter_context([f"{w['title']} {w['snippet']}" for w in web])
        web = [w for w, ok in zip(web, web_safe) if ok]

        blocks: list[str] = []
        for i, doc in enumerate(docs, start=1):
            blocks.append(
                f"[K{i}] source={doc['source']} | {doc['location']} | type={doc['modality']}\n{doc['text']}"
            )
        for j, item in enumerate(web, start=1):
            blocks.append(f"[W{j}] {item['title']} ({item['url']})\n{item['snippet']}")

        images = self._image_data_urls(docs)
        context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
        user = f"Question: {state['question']}\n\n<context>\n{context}\n</context>"
        if state.get("guard_feedback"):
            user += (
                "\n\nA quality check rejected your previous draft: "
                f"{state['guard_feedback']}\nRewrite it using ONLY facts present in <context>; remove or "
                "flag anything unsupported."
            )

        history = [
            {"role": "user" if m.type == "human" else "assistant", "content": truncate(str(m.content), 1500)}
            for m in state.get("messages", [])[:-1][-settings.history_messages:]
        ]
        text = self.gateway.complete(
            "answer", ANSWER_SYSTEM, user, history=history, images=images or None,
            temperature=0.2, max_tokens=1800,
        )
        return {
            "answer": text,
            "contexts": [b.split("\n", 1)[-1] for b in blocks],
            "relevant_docs": docs,
            "web_results": web,
            "route_log": state.get("route_log", []) + ["answer"],
        }

    def _image_data_urls(self, docs: list[dict[str, Any]], limit: int = 2) -> list[str]:
        ids = [d["id"] for d in docs if d.get("modality") == "image"][:limit]
        if not ids:
            return []
        try:
            fetched = self.store.fetch(ids)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not fetch image thumbnails: %s", exc)
            return []
        return [
            f"data:{meta.get('image_mime', 'image/jpeg')};base64,{meta['image_b64']}"
            for meta in fetched.values()
            if meta.get("image_b64")
        ]

    # ------------------------------------------------------------------ #
    def finalize(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        cited = set(re.findall(r"\[([KW]\d+)\]", answer))
        lines: list[str] = []
        for i, doc in enumerate(state.get("relevant_docs", []), start=1):
            if f"K{i}" in cited:
                where = f", {doc['location']}" if doc.get("location") and doc["location"] != "body" else ""
                lines.append(f"- [K{i}] {doc['source']}{where}")
        for j, item in enumerate(state.get("web_results", []), start=1):
            if f"W{j}" in cited:
                lines.append(f"- [W{j}] {item['title']} \u2013 {item['url']}")
        final = answer + ("\n\n**Sources**\n" + "\n".join(lines) if lines else "")
        return {"final_answer": final, "messages": [AIMessage(content=answer)]}
