"""Top-level LangGraph: input guard -> supervisor <-> sub-agents -> answer -> output guard -> finalize."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.crag import build_crag_graph
from src.agents.nodes import AgentNodes
from src.agents.state import AgentState


def build_graph(gateway, guardrails, store, retriever, reranker, kb_overview):
    nodes = AgentNodes(gateway, guardrails, store, kb_overview)
    crag = build_crag_graph(gateway, retriever, reranker)  # the retrieval sub-agent

    graph = StateGraph(AgentState)
    graph.add_node("input_guard", nodes.input_guard)
    graph.add_node("supervisor", nodes.supervisor)
    graph.add_node("retrieval_agent", crag)
    graph.add_node("web_search_agent", nodes.web_search_agent)
    graph.add_node("answer_agent", nodes.answer_agent)
    graph.add_node("output_guard", nodes.output_guard)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard", nodes.route_after_input_guard, {"supervisor": "supervisor", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "supervisor",
        nodes.route_from_supervisor,
        {
            "retrieval_agent": "retrieval_agent",
            "web_search_agent": "web_search_agent",
            "answer_agent": "answer_agent",
        },
    )
    graph.add_edge("retrieval_agent", "supervisor")
    graph.add_edge("web_search_agent", "supervisor")
    graph.add_edge("answer_agent", "output_guard")
    graph.add_conditional_edges(
        "output_guard", nodes.route_after_output_guard, {"answer_agent": "answer_agent", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())
