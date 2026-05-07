# agent/graph.py

import os
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent.state import AgentState
from agent.nodes import AgentNodes
from agent.tools import RAGTool, WebSearchTool
from core.generation import Generator
from observability.tracing import TracingClient
from entity.config_entity import GeneratorConfig
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def route_decision(state: AgentState):
    route = state.get("route", "rag")

    if route == "both":
        return [
            Send("rag_node", state),
            Send("web_node", state),
        ]
    elif route == "web":
        return Send("web_node", state)
    else:
        return Send("rag_node", state)


def build_graph(nodes: AgentNodes) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router_node", nodes.router_node)
    graph.add_node("rag_node", nodes.rag_node)
    graph.add_node("web_node", nodes.web_node)
    graph.add_node("combine_node", nodes.combine_node)
    graph.add_node("answer_node", nodes.answer_node)

    graph.add_edge(START, "router_node")

    # no third argument — Send already carries the destination
    graph.add_conditional_edges("router_node", route_decision)

    graph.add_edge("rag_node", "combine_node")
    graph.add_edge("web_node", "combine_node")
    graph.add_edge("combine_node", "answer_node")
    graph.add_edge("answer_node", END)

    compiled = graph.compile()
    logger.info("Graph compiled — router → [rag | web | both] → combine → answer")
    return compiled


def create_agent():
    """Build all dependencies and return compiled graph + tracer. Call once at startup."""
    rag_tool = RAGTool()
    web_tool = WebSearchTool()
    generator = Generator(
        GeneratorConfig(
            groq_api_key=os.environ["GROQ_API_KEY"],
            model=settings.GROQ_MODEL,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
        )
    )
    tracer = TracingClient()
    nodes = AgentNodes(rag_tool, web_tool, generator, tracer)
    graph = build_graph(nodes)
    return graph, tracer


def run_agent(
    graph,
    tracer: TracingClient,
    question: str,
    chat_history: list = None,
) -> dict:
    """
    Run one question through the graph and return the final state.
    chat_history is a list of {"role": ..., "content": ...} dicts.
    """
    trace_id = tracer.generate_trace_id()
    tracer.start_trace(trace_id, question)

    initial_state: AgentState = {
        "question": question,
        "route": None,
        "rag_results": None,
        "web_results": None,
        "combined_context": None,
        "answer": None,
        "sources": None,
        "chat_history": chat_history or [],
        "trace_id": trace_id,
    }

    logger.info(f"Running agent: '{question[:80]}'")
    final_state = graph.invoke(initial_state)
    logger.info(
        f"Agent done — route={final_state['route']}, "
        f"answer={len(final_state.get('answer', ''))} chars"
    )

    # flush after invoke so all spans are sent before the response returns
    tracer.end_trace(trace_id)

    return final_state