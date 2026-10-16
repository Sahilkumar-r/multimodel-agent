from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State shared by the supervisor, the sub-agents and the C-RAG sub-graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    question: str

    # supervisor
    next_agent: str
    supervisor_reason: str
    steps: int
    route_log: list[str]

    # retrieval_agent (C-RAG)
    rewritten_queries: list[str]
    candidates: list[dict[str, Any]]
    ranked: list[dict[str, Any]]
    relevant_docs: list[dict[str, Any]]
    crag_verdict: str  # correct | ambiguous | incorrect
    retrieval_done: bool

    # web_search_agent
    web_results: list[dict[str, str]]
    web_done: bool

    # answer + guardrails
    contexts: list[str]
    answer: str
    final_answer: str
    guard_feedback: str
    retry_answer: bool
    retries: int
    blocked: bool
