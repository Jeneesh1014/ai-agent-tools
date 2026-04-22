"""
Manual smoke test for router_node, rag_node, web_node.
Run from project root: python test_nodes_manual.py

RAGTool will skip re-embedding if chroma_db/ already exists from Day 3.
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from utils.logger import get_logger

logger = get_logger("test_nodes")


def build_nodes():
    """Build all dependencies once and return an AgentNodes instance."""
    from agent.tools import RAGTool, WebSearchTool
    from core.generation import Generator
    from agent.nodes import AgentNodes
    from entity.config_entity import GeneratorConfig
    from config import settings
    import os

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
    return AgentNodes(rag_tool, web_tool, generator)


def test_router_rag(nodes):
    logger.info("--- Router test: expects 'rag' ---")
    from agent.state import AgentState

    state: AgentState = {
        "question": "What is the attention mechanism in transformers?",
        "route": None,
        "rag_results": None,
        "web_results": None,
        "combined_context": None,
        "answer": None,
        "sources": None,
        "chat_history": [],
        "trace_id": None,
    }

    result = nodes.router_node(state)
    print(f"  Route: {result['route']}")
    assert result["route"] == "rag", f"Expected 'rag', got '{result['route']}'"
    logger.info("Router → rag PASS")


def test_router_web(nodes):
    logger.info("--- Router test: expects 'web' ---")
    from agent.state import AgentState

    state: AgentState = {
        "question": "What LLM did Google release last week?",
        "route": None,
        "rag_results": None,
        "web_results": None,
        "combined_context": None,
        "answer": None,
        "sources": None,
        "chat_history": [],
        "trace_id": None,
    }

    result = nodes.router_node(state)
    print(f"  Route: {result['route']}")
    assert result["route"] == "web", f"Expected 'web', got '{result['route']}'"
    logger.info("Router → web PASS")


def test_router_both(nodes):
    logger.info("--- Router test: expects 'both' ---")
    from agent.state import AgentState

    state: AgentState = {
        "question": "How does my research on attention mechanisms compare to what Google released recently?",
        "route": None,
        "rag_results": None,
        "web_results": None,
        "combined_context": None,
        "answer": None,
        "sources": None,
        "chat_history": [],
        "trace_id": None,
    }

    result = nodes.router_node(state)
    print(f"  Route: {result['route']}")
    assert result["route"] == "both", f"Expected 'both', got '{result['route']}'"
    logger.info("Router → both PASS")


def test_rag_node(nodes):
    logger.info("--- RAG node test ---")
    from agent.state import AgentState

    state: AgentState = {
        "question": "Explain self-attention in transformer models",
        "route": "rag",
        "rag_results": None,
        "web_results": None,
        "combined_context": None,
        "answer": None,
        "sources": None,
        "chat_history": [],
        "trace_id": None,
    }

    result = nodes.rag_node(state)
    rag_results = result["rag_results"]

    assert rag_results is not None
    assert len(rag_results) > 0
    assert "content" in rag_results[0]
    assert "source" in rag_results[0]
    assert "relevance_score" in rag_results[0]

    for r in rag_results:
        print(f"  [{r['relevance_score']:.3f}] {r['source']} — {r['content'][:80]}...")

    logger.info(f"RAG node PASS — {len(rag_results)} results")


def test_web_node(nodes):
    logger.info("--- Web node test ---")
    from agent.state import AgentState

    state: AgentState = {
        "question": "Latest AI research papers 2024",
        "route": "web",
        "rag_results": None,
        "web_results": None,
        "combined_context": None,
        "answer": None,
        "sources": None,
        "chat_history": [],
        "trace_id": None,
    }

    result = nodes.web_node(state)
    web_results = result["web_results"]

    assert web_results is not None
    assert len(web_results) > 0
    assert "content" in web_results[0]
    assert "url" in web_results[0]
    assert "title" in web_results[0]

    for r in web_results:
        print(f"  [{r['score']:.3f}] {r['title']} — {r['url']}")

    logger.info(f"Web node PASS — {len(web_results)} results")


if __name__ == "__main__":
    failures = []

    logger.info("Building tools — RAG index loads from disk if Day 3 ran already")
    try:
        nodes = build_nodes()
    except Exception as e:
        logger.error(f"Failed to build nodes: {e}")
        sys.exit(1)

    tests = [
        ("router → rag", test_router_rag),
        ("router → web", test_router_web),
        ("router → both", test_router_both),
        ("rag node", test_rag_node),
        ("web node", test_web_node),
    ]

    for name, fn in tests:
        try:
            fn(nodes)
            print(f"{name}: PASS")
        except Exception as e:
            logger.error(f"{name}: FAIL — {e}")
            failures.append(name)

    print()
    if failures:
        print(f"Failed: {failures}")
        sys.exit(1)
    else:
        print("All Day 4 tests passed")