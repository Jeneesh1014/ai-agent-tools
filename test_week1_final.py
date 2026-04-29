"""
Week 1 final verification — 5 questions, all three routes tested.
Run from project root: python test_week1_final.py
"""

import sys
import time
from dotenv import load_dotenv

load_dotenv()

from utils.logger import get_logger

logger = get_logger("week1_final")

QUESTIONS = [
    {
        "question": "What problem does the LoRA paper solve and how?",
        "expected_route": "rag",
        "note": "LoRA is in the documents — should go straight to RAG",
    },
    {
        "question": "What AI models has Anthropic released in 2025?",
        "expected_route": "web",
        "note": "Recent releases not in any paper — web only",
    },
    {
        "question": "How does BERT differ from GPT in terms of pretraining objectives?",
        "expected_route": "rag",
        "note": "Both papers are in the collection",
    },
    {
        "question": "What are the current state of the art benchmark scores for LLMs in 2024?",
        "expected_route": "web",
        "note": "Benchmark leaderboards are live data — needs web",
    },
    {
        "question": "How does the vision transformer (ViT) architecture work and how do recent vision models build on it?",
        "expected_route": "both",
        "note": "ViT paper is in docs, recent models need web",
    },
]


if __name__ == "__main__":
    logger.info("Week 1 final verification starting")

    from agent.graph import create_agent, run_agent

    logger.info("Building graph")
    start = time.time()
    graph = create_agent()
    logger.info(f"Graph ready in {time.time() - start:.1f}s")

    failures = []
    route_mismatches = []

    for i, item in enumerate(QUESTIONS, 1):
        question = item["question"]
        expected = item["expected_route"]
        note = item["note"]

        logger.info(f"--- Question {i}/5 ---")
        logger.info(f"  {question}")
        logger.info(f"  Expected route: {expected} ({note})")

        try:
            t0 = time.time()
            state = run_agent(graph, question)
            elapsed = time.time() - t0

            actual_route = state["route"]
            answer = state["answer"]
            sources = state["sources"]

            assert answer, "No answer returned"
            assert len(answer) > 30, "Answer suspiciously short"
            assert sources, "No sources returned"

            route_ok = actual_route == expected
            if not route_ok:
                route_mismatches.append({
                    "q": question[:60],
                    "expected": expected,
                    "actual": actual_route,
                })

            print(f"\nQ{i}: {question[:70]}...")
            print(f"  Route:    {actual_route} {'✓' if route_ok else f'(expected {expected})'}")
            print(f"  Sources:  {len(sources)} — {sources[0][:60]}")
            print(f"  Answer:   {answer[:200]}...")
            print(f"  Time:     {elapsed:.1f}s")
            print(f"  Status:   PASS")

        except Exception as e:
            logger.error(f"Q{i} FAIL — {e}")
            failures.append(f"Q{i}: {question[:50]}")

    print("\n" + "="*60)
    print("WEEK 1 FINAL RESULTS")
    print("="*60)
    print(f"Answered:  {len(QUESTIONS) - len(failures)}/{len(QUESTIONS)}")
    print(f"Failures:  {len(failures)}")
    print(f"Route mismatches: {len(route_mismatches)}")

    if route_mismatches:
        print("\nRoute mismatches (not failures — routing improves in Week 3):")
        for m in route_mismatches:
            print(f"  '{m['q']}' → got {m['actual']}, expected {m['expected']}")

    if failures:
        print(f"\nActual failures: {failures}")
        sys.exit(1)
    else:
        print("\nAll questions answered. Week 1 complete.")