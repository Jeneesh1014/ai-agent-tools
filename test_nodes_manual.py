"""
Manual smoke test for combine_node and answer_node.
Run from project root: python test_nodes_day5_manual.py
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from utils.logger import get_logger

logger = get_logger("test_nodes_day5")


def build_nodes():
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


def make_state(question, route, rag_results=None, web_results=None):
    return {
        "question": question,
        "route": route,
        "rag_results": rag_results,
        "web_results": web_results,
        "combined_context": None,
        "answer": None,
        "sources": None,
        "chat_history": [],
        "trace_id": None,
    }


def test_combine_rag_only(nodes):
    logger.info("--- combine_node: rag only ---")

    state = make_state(
        question="What is self-attention?",
        route="rag",
        rag_results=[
            {"content": "Self-attention allows each position to attend to all positions.", "source": "transformer.pdf", "relevance_score": 0.95},
            {"content": "The attention mechanism computes query, key, value projections.", "source": "transformer.pdf", "relevance_score": 0.88},
        ],
    )

    result = nodes.combine_node(state)

    assert result["combined_context"]
    assert "Research Documents" in result["combined_context"]
    assert "Web Search Results" not in result["combined_context"]
    assert len(result["sources"]) == 1  # deduped — same file twice
    print(f"  Context length: {len(result['combined_context'])} chars")
    print(f"  Sources: {result['sources']}")
    logger.info("combine_node rag only PASS")


def test_combine_web_only(nodes):
    logger.info("--- combine_node: web only ---")

    state = make_state(
        question="Latest AI news?",
        route="web",
        web_results=[
            {"content": "OpenAI released GPT-5.", "url": "https://example.com/1", "title": "GPT-5 news", "score": 0.9},
            {"content": "Google released Gemini Ultra 2.", "url": "https://example.com/2", "title": "Gemini news", "score": 0.85},
        ],
    )

    result = nodes.combine_node(state)

    assert result["combined_context"]
    assert "Web Search Results" in result["combined_context"]
    assert "Research Documents" not in result["combined_context"]
    assert len(result["sources"]) == 2
    print(f"  Context length: {len(result['combined_context'])} chars")
    print(f"  Sources: {result['sources']}")
    logger.info("combine_node web only PASS")


def test_combine_both(nodes):
    logger.info("--- combine_node: both ---")

    state = make_state(
        question="Compare attention in papers vs latest models",
        route="both",
        rag_results=[
            {"content": "Attention is all you need paper content.", "source": "transformer.pdf", "relevance_score": 0.93},
        ],
        web_results=[
            {"content": "New transformer variants released in 2024.", "url": "https://example.com/3", "title": "New Models", "score": 0.88},
        ],
    )

    result = nodes.combine_node(state)

    assert result["combined_context"]
    assert "Research Documents" in result["combined_context"]
    assert "Web Search Results" in result["combined_context"]
    assert len(result["sources"]) == 2
    print(f"  Context length: {len(result['combined_context'])} chars")
    print(f"  Sources: {result['sources']}")
    logger.info("combine_node both PASS")


def test_answer_node_rag(nodes):
    logger.info("--- answer_node: rag route ---")

    # run rag_node to get real results, then combine, then answer
    state = make_state(
        question="What is self-attention in transformers?",
        route="rag",
    )

    state.update(nodes.rag_node(state))
    state.update(nodes.combine_node(state))
    result = nodes.answer_node(state)

    assert result["answer"]
    assert len(result["answer"]) > 50
    assert result["chat_history"]
    assert result["chat_history"][0]["role"] == "user"
    assert result["chat_history"][1]["role"] == "assistant"

    print(f"\n  Question: {state['question']}")
    print(f"  Answer:   {result['answer'][:300]}...")
    print(f"  History entries added: {len(result['chat_history'])}")
    logger.info("answer_node rag PASS")


def test_answer_node_web(nodes):
    logger.info("--- answer_node: web route ---")

    state = make_state(
        question="What are the latest LLM releases in 2024?",
        route="web",
    )

    state.update(nodes.web_node(state))
    state.update(nodes.combine_node(state))
    result = nodes.answer_node(state)

    assert result["answer"]
    assert len(result["answer"]) > 50

    print(f"\n  Question: {state['question']}")
    print(f"  Answer:   {result['answer'][:300]}...")
    logger.info("answer_node web PASS")


def test_full_pipeline_both(nodes):
    logger.info("--- Full pipeline: both route ---")

    state = make_state(
        question="How does the transformer attention mechanism compare to recent model architectures?",
        route=None,
    )

    # run all nodes in sequence — graph will do this automatically on Day 6
    state.update(nodes.router_node(state))
    logger.info(f"  Route decided: {state['route']}")

    if state["route"] in ("rag", "both"):
        state.update(nodes.rag_node(state))
    if state["route"] in ("web", "both"):
        state.update(nodes.web_node(state))

    state.update(nodes.combine_node(state))
    state.update(nodes.answer_node(state))

    assert state["answer"]
    assert state["sources"]
    assert len(state["chat_history"]) == 2

    print(f"\n  Route:    {state['route']}")
    print(f"  Sources:  {state['sources']}")
    print(f"  Answer:   {state['answer'][:400]}...")
    logger.info("Full pipeline PASS")


if __name__ == "__main__":
    failures = []

    logger.info("Building tools")
    try:
        nodes = build_nodes()
    except Exception as e:
        logger.error(f"Failed to build nodes: {e}")
        sys.exit(1)

    tests = [
        ("combine rag only", test_combine_rag_only),
        ("combine web only", test_combine_web_only),
        ("combine both", test_combine_both),
        ("answer rag", test_answer_node_rag),
        ("answer web", test_answer_node_web),
        ("full pipeline", test_full_pipeline_both),
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
        print("All Day 5 tests passed")