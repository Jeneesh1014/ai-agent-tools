"""
End-to-end test of the compiled LangGraph agent.
Run from project root: python test_graph_manual.py

Tests 5 different questions covering all three routes.
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from utils.logger import get_logger

logger = get_logger("test_graph")


def test_rag_question(graph):
    logger.info("--- Test 1: RAG question ---")
    from agent.graph import run_agent

    state = run_agent(graph, "What is the attention mechanism in transformers?")

    assert state["answer"], "No answer returned"
    assert state["route"] == "rag", f"Expected rag, got {state['route']}"
    assert state["sources"], "No sources returned"
    assert state["rag_results"], "No RAG results"
    assert state["web_results"] is None or state["web_results"] == []

    print(f"\n  Route:   {state['route']}")
    print(f"  Sources: {state['sources']}")
    print(f"  Answer:  {state['answer'][:250]}...")
    logger.info("Test 1 PASS")


def test_web_question(graph):
    logger.info("--- Test 2: Web question ---")
    from agent.graph import run_agent

    state = run_agent(graph, "What LLM did Google release last week?")

    assert state["answer"], "No answer returned"
    assert state["route"] == "web", f"Expected web, got {state['route']}"
    assert state["sources"], "No sources returned"
    assert state["web_results"], "No web results"

    print(f"\n  Route:   {state['route']}")
    print(f"  Sources: {state['sources'][:3]}")
    print(f"  Answer:  {state['answer'][:250]}...")
    logger.info("Test 2 PASS")


def test_both_question(graph):
    logger.info("--- Test 3: Both route question ---")
    from agent.graph import run_agent

    state = run_agent(
        graph,
        "How does the attention mechanism from the original transformer paper compare to how modern LLMs use it today?"
    )

    assert state["answer"], "No answer returned"
    assert state["route"] == "both", f"Expected both, got {state['route']}"
    assert state["rag_results"], "No RAG results"
    assert state["web_results"], "No web results"
    assert state["sources"], "No sources"

    print(f"\n  Route:   {state['route']}")
    print(f"  RAG results:  {len(state['rag_results'])}")
    print(f"  Web results:  {len(state['web_results'])}")
    print(f"  Answer:  {state['answer'][:250]}...")
    logger.info("Test 3 PASS")


def test_chat_history_carries(graph):
    logger.info("--- Test 4: Chat history carries across turns ---")
    from agent.graph import run_agent

    # first turn
    state1 = run_agent(graph, "What is LoRA fine-tuning?")
    history = state1["chat_history"]
    assert len(history) == 2, f"Expected 2 history entries, got {len(history)}"

    # second turn references the first
    state2 = run_agent(graph, "How does it compare to full fine-tuning?", chat_history=history)
    assert state2["answer"], "No answer on second turn"

    # history should now have 4 entries — 2 from each turn
    assert len(state2["chat_history"]) == 4, f"Expected 4 history entries, got {len(state2['chat_history'])}"

    print(f"\n  Turn 1 answer: {state1['answer'][:150]}...")
    print(f"  Turn 2 answer: {state2['answer'][:150]}...")
    print(f"  History length after 2 turns: {len(state2['chat_history'])}")
    logger.info("Test 4 PASS")


def test_sources_always_present(graph):
    logger.info("--- Test 5: Sources always non-empty ---")
    from agent.graph import run_agent

    questions = [
        "Explain BERT pretraining",
        "What is the latest version of GPT?",
    ]

    for q in questions:
        state = run_agent(graph, q)
        assert state["sources"], f"No sources for: {q}"
        assert state["answer"], f"No answer for: {q}"
        print(f"\n  Q: {q}")
        print(f"  Route: {state['route']}")
        print(f"  Sources: {state['sources'][:2]}")

    logger.info("Test 5 PASS")


if __name__ == "__main__":
    failures = []

    logger.info("Building graph — RAG index takes ~80s on first run")
    try:
        from agent.graph import create_agent
        graph = create_agent()
        logger.info("Graph ready")
    except Exception as e:
        logger.error(f"Graph build failed: {e}")
        sys.exit(1)

    tests = [
        ("rag question", test_rag_question),
        ("web question", test_web_question),
        ("both question", test_both_question),
        ("chat history carries", test_chat_history_carries),
        ("sources always present", test_sources_always_present),
    ]

    for name, fn in tests:
        try:
            fn(graph)
            print(f"\n{name}: PASS")
        except Exception as e:
            logger.error(f"{name}: FAIL — {e}")
            failures.append(name)

    print()
    if failures:
        print(f"Failed: {failures}")
        sys.exit(1)
    else:
        print("All Day 6 tests passed — end-to-end graph working")