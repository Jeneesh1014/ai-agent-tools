import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import create_agent, run_agent  # noqa: E402
from config import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

BASELINE_OUTPUT = settings.BASE_DIR / "reports" / "evaluation_results.json"


@dataclass
class EvaluationCase:
    question: str
    expected_route: str
    reason: str


@dataclass
class EvaluationResult:
    question: str
    expected_route: str
    predicted_route: str
    correct: bool
    mode: str
    latency_seconds: float
    answer_preview: str


EVALUATION_CASES: List[EvaluationCase] = [
    EvaluationCase(
        question="What is the attention mechanism in transformers?",
        expected_route="rag",
        reason="Foundational architecture question from research papers.",
    ),
    EvaluationCase(
        question="Explain how LoRA reduces trainable parameters.",
        expected_route="rag",
        reason="Method explanation belongs in internal research documents.",
    ),
    EvaluationCase(
        question="What is retrieval augmented generation?",
        expected_route="rag",
        reason="Conceptual RAG question should use the paper collection.",
    ),
    EvaluationCase(
        question="How do convolutional neural networks process images?",
        expected_route="rag",
        reason="Stable deep learning concept from documents.",
    ),
    EvaluationCase(
        question="What AI model did Google release last week?",
        expected_route="web",
        reason="Recent release requires current web information.",
    ),
    EvaluationCase(
        question="What are the latest LLM benchmark results this month?",
        expected_route="web",
        reason="Current benchmarks change over time.",
    ),
    EvaluationCase(
        question="Who is currently leading OpenAI?",
        expected_route="web",
        reason="Current leadership information can change.",
    ),
    EvaluationCase(
        question="Compare the transformer paper with the latest long-context LLM work.",
        expected_route="both",
        reason="Needs internal paper context and current web information.",
    ),
    EvaluationCase(
        question="How does LoRA compare with recent parameter-efficient fine-tuning methods?",
        expected_route="both",
        reason="Needs LoRA paper context plus newer methods.",
    ),
    EvaluationCase(
        question="How does my RAG research compare to the newest agentic RAG systems?",
        expected_route="both",
        reason="Needs local RAG knowledge and recent agent work.",
    ),
]


def heuristic_route(question: str) -> str:
    text = question.lower()
    current_terms = [
        "latest",
        "last week",
        "this month",
        "currently",
        "newest",
        "recent",
        "today",
    ]
    research_terms = [
        "attention",
        "transformer",
        "lora",
        "rag",
        "retrieval augmented generation",
        "convolutional",
        "paper",
        "research",
        "fine-tuning",
    ]
    asks_current = any(term in text for term in current_terms)
    asks_research = any(term in text for term in research_terms)

    if asks_current and asks_research:
        return "both"
    if asks_current:
        return "web"
    return "rag"


def evaluate_mock() -> List[EvaluationResult]:
    results = []
    for case in EVALUATION_CASES:
        start = time.monotonic()
        predicted = heuristic_route(case.question)
        elapsed = time.monotonic() - start
        results.append(
            EvaluationResult(
                question=case.question,
                expected_route=case.expected_route,
                predicted_route=predicted,
                correct=predicted == case.expected_route,
                mode="mock",
                latency_seconds=elapsed,
                answer_preview="Mock routing-only evaluation; run with --live for full agent answers.",
            )
        )
    return results


def evaluate_live() -> List[EvaluationResult]:
    required = ["GROQ_API_KEY", "COHERE_API_KEY", "TAVILY_API_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    graph, tracer = create_agent()
    results = []

    for case in EVALUATION_CASES:
        start = time.monotonic()
        final_state = run_agent(graph, tracer, question=case.question, chat_history=[])
        elapsed = time.monotonic() - start
        predicted = final_state.get("route") or "rag"
        answer = final_state.get("answer") or ""
        results.append(
            EvaluationResult(
                question=case.question,
                expected_route=case.expected_route,
                predicted_route=predicted,
                correct=predicted == case.expected_route,
                mode="live",
                latency_seconds=elapsed,
                answer_preview=answer[:240],
            )
        )

    return results


def write_report(results: List[EvaluationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct = sum(1 for result in results if result.correct)
    payload = {
        "project": "Intelligent Research Agent",
        "model": settings.GROQ_MODEL,
        "total_cases": len(results),
        "correct": correct,
        "routing_accuracy": correct / len(results) if results else 0.0,
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        f"Evaluation complete — {correct}/{len(results)} correct, "
        f"saved to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate router accuracy.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the real LangGraph agent. This uses API calls.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASELINE_OUTPUT,
        help="Where to save the JSON report.",
    )
    args = parser.parse_args()

    results = evaluate_live() if args.live else evaluate_mock()
    write_report(results, args.output)


if __name__ == "__main__":
    main()
