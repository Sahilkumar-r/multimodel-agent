"""Corrective RAG (C-RAG) sub-graph, used as the supervisor's ``retrieval_agent``.

    rewrite_query -> retrieve (hybrid + RRF) -> rerank (bi-encoder) -> grade (LLM)
        correct   -> refine ----------------------------------------> done
        ambiguous -> refine -> web_fallback (DuckDuckGo) ------------> done
        incorrect -> web_fallback (DuckDuckGo) ----------------------> done
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.state import AgentState
from src.agents.tools import call_tool
from src.config import settings
from src.prompts import GRADER_SYSTEM, QUERY_REWRITER_SYSTEM
from src.utils import dedupe, truncate

log = logging.getLogger(__name__)


def format_history(messages: list[Any], limit: int) -> str:
    lines = []
    for message in messages[-limit:]:
        role = "User" if message.type == "human" else "Assistant"
        lines.append(f"{role}: {truncate(str(message.content), 500)}")
    return "\n".join(lines) or "(no earlier messages)"


def build_crag_graph(gateway, retriever, reranker):
    def rewrite_query(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        history = format_history(state.get("messages", [])[:-1], settings.history_messages)
        queries = [question]
        try:
            data = gateway.complete_json(
                "query_rewriter",
                QUERY_REWRITER_SYSTEM,
                f"Conversation so far:\n{history}\n\nLatest user message: {question}",
                max_tokens=400,
            )
            standalone = str(data.get("standalone_query") or question).strip()
            variants = [v for v in data.get("variants", []) if isinstance(v, str)]
            queries = dedupe([standalone] + variants[: settings.query_variants])
        except Exception as exc:  # noqa: BLE001
            log.warning("Query rewrite failed, using the raw question: %s", exc)
        return {"rewritten_queries": queries, "route_log": state.get("route_log", []) + ["rewrite"]}

    def retrieve(state: AgentState) -> dict[str, Any]:
        found = retriever.search(state["rewritten_queries"])
        return {
            "candidates": [c.to_doc() for c in found],
            "route_log": state.get("route_log", []) + [f"hybrid_rrf({len(found)})"],
        }

    def rerank(state: AgentState) -> dict[str, Any]:
        docs = state.get("candidates", [])
        if not docs:
            return {"ranked": []}
        query = state["rewritten_queries"][0]
        scores = reranker.score(query, [d["text"] for d in docs])
        scored = [dict(d, rerank_score=round(float(s), 4)) for d, s in zip(docs, scores)]
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return {
            "ranked": scored[: settings.rerank_top_n],
            "route_log": state.get("route_log", []) + ["rerank"],
        }

    def grade(state: AgentState) -> dict[str, Any]:
        docs = state.get("ranked", [])
        if not docs:
            return {"relevant_docs": [], "crag_verdict": "incorrect",
                    "route_log": state.get("route_log", []) + ["grade:empty-kb"]}

        passages = "\n\n".join(
            f"[{i}] ({d['modality']}, {d['source']}, {d['location']})\n{truncate(d['text'], settings.grade_max_chars)}"
            for i, d in enumerate(docs)
        )
        relevant_idx: set[int] | None = None
        try:
            data = gateway.complete_json(
                "grader",
                GRADER_SYSTEM,
                f"Question: {state['rewritten_queries'][0]}\n\nPassages:\n{passages}",
                max_tokens=500,
            )
            relevant_idx = {
                int(g["index"]) for g in data.get("grades", []) if g.get("relevant") is True and "index" in g
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM grading failed, falling back to the re-ranker score: %s", exc)
        if relevant_idx is None:
            relevant_idx = {
                i for i, d in enumerate(docs) if d["rerank_score"] >= settings.rerank_fallback_threshold
            }

        relevant = [d for i, d in enumerate(docs) if i in relevant_idx]
        if len(relevant) >= settings.crag_min_relevant:
            verdict = "correct"
        elif not relevant:
            verdict = "incorrect"
        else:
            verdict = "ambiguous"
        return {
            "relevant_docs": relevant,
            "crag_verdict": verdict,
            "route_log": state.get("route_log", []) + [f"grade:{verdict}({len(relevant)}/{len(docs)})"],
        }

    def refine(state: AgentState) -> dict[str, Any]:
        query = state["rewritten_queries"][0]
        refined = []
        for doc in state.get("relevant_docs", []):
            if doc["modality"] == "image":
                refined.append(doc)
            else:
                refined.append(dict(doc, text=reranker.refine(query, doc["text"])))
        return {"relevant_docs": refined, "route_log": state.get("route_log", []) + ["refine"]}

    def web_fallback(state: AgentState) -> dict[str, Any]:
        query = state["rewritten_queries"][0]
        results = call_tool("duckduckgo_search", query=query, max_results=settings.web_max_results)
        merged = {r["url"]: r for r in state.get("web_results", []) + results}
        return {
            "web_results": list(merged.values()),
            "web_done": True,
            "route_log": state.get("route_log", []) + [f"web_fallback({len(results)})"],
        }

    def mark_done(state: AgentState) -> dict[str, Any]:
        return {"retrieval_done": True}

    def after_grade(state: AgentState) -> str:
        return "web_fallback" if state.get("crag_verdict") == "incorrect" else "refine"

    def after_refine(state: AgentState) -> str:
        return "web_fallback" if state.get("crag_verdict") == "ambiguous" else "mark_done"

    graph = StateGraph(AgentState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("grade", grade)
    graph.add_node("refine", refine)
    graph.add_node("web_fallback", web_fallback)
    graph.add_node("mark_done", mark_done)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade")
    graph.add_conditional_edges("grade", after_grade, {"web_fallback": "web_fallback", "refine": "refine"})
    graph.add_conditional_edges("refine", after_refine, {"web_fallback": "web_fallback", "mark_done": "mark_done"})
    graph.add_edge("web_fallback", "mark_done")
    graph.add_edge("mark_done", END)
    return graph.compile()
