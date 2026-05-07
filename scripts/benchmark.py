import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import create_agent, run_agent  # noqa: E402
from config import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

BENCHMARK_OUTPUT = settings.BASE_DIR / "reports" / "benchmark_results.json"

BENCHMARK_QUERIES = [
    "What is the attention mechanism?",
    "Explain RAG in simple terms.",
    "What AI model did Google release recently?",
    "Compare LoRA with the latest fine-tuning methods.",
    "How do transformers use positional encoding?",
    "What are the newest long-context LLM techniques?",
    "How does CLIP connect text and images?",
    "Compare BERT pretraining with current embedding models.",
    "What is BM25?",
    "How does my RAG pipeline compare with recent agentic RAG systems?",
    "Explain residual connections.",
    "What are current open-source LLM trends?",
    "How does Stable Diffusion generate images?",
    "Compare CNNs with modern vision transformers.",
    "What is chain-of-thought prompting?",
    "What changed in AI regulation recently?",
    "Explain reinforcement learning from human feedback.",
    "How does Mistral compare with newer open models?",
    "What is Word2Vec?",
    "Compare GPT-3 with the latest frontier models.",
]


@dataclass
class BenchmarkRow:
    question: str
    route: str
    latency_seconds: float
    rag_results_count: int
    web_results_count: int
    mode: str


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * pct)
    return sorted_values[index]


def summarize(rows: List[BenchmarkRow]) -> Dict:
    latencies = [row.latency_seconds for row in rows]
    by_route: Dict[str, List[float]] = {}
    for row in rows:
        by_route.setdefault(row.route, []).append(row.latency_seconds)

    return {
        "total_queries": len(rows),
        "p50_latency_seconds": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_seconds": percentile(latencies, 0.95),
        "avg_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
        "by_route": {
            route: {
                "count": len(route_latencies),
                "p50_latency_seconds": statistics.median(route_latencies),
                "p95_latency_seconds": percentile(route_latencies, 0.95),
            }
            for route, route_latencies in by_route.items()
        },
    }


def mock_route(question: str) -> str:
    text = question.lower()
    current_terms = ["latest", "recent", "current", "newer", "newest"]
    research_terms = [
        "rag",
        "lora",
        "bert",
        "gpt-3",
        "clip",
        "transformer",
        "mistral",
    ]
    asks_current = any(term in text for term in current_terms)
    asks_research = any(term in text for term in research_terms)
    if asks_current and asks_research:
        return "both"
    if asks_current:
        return "web"
    return "rag"


def benchmark_mock() -> List[BenchmarkRow]:
    rows = []
    for query in BENCHMARK_QUERIES:
        start = time.monotonic()
        route = mock_route(query)
        elapsed = time.monotonic() - start
        rows.append(
            BenchmarkRow(
                question=query,
                route=route,
                latency_seconds=elapsed,
                rag_results_count=1 if route in {"rag", "both"} else 0,
                web_results_count=1 if route in {"web", "both"} else 0,
                mode="mock",
            )
        )
    return rows


def benchmark_live() -> List[BenchmarkRow]:
    required = ["GROQ_API_KEY", "COHERE_API_KEY", "TAVILY_API_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    graph, tracer = create_agent()
    rows = []

    for query in BENCHMARK_QUERIES:
        start = time.monotonic()
        final_state = run_agent(graph, tracer, question=query, chat_history=[])
        elapsed = time.monotonic() - start
        route = final_state.get("route") or "rag"
        rows.append(
            BenchmarkRow(
                question=query,
                route=route,
                latency_seconds=elapsed,
                rag_results_count=len(final_state.get("rag_results") or []),
                web_results_count=len(final_state.get("web_results") or []),
                mode="live",
            )
        )

    return rows


def write_report(rows: List[BenchmarkRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project": "Intelligent Research Agent",
        "model": settings.GROQ_MODEL,
        "summary": summarize(rows),
        "results": [asdict(row) for row in rows],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Benchmark complete — saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark agent latency.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the real LangGraph agent. This uses API calls.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_OUTPUT,
        help="Where to save the JSON report.",
    )
    args = parser.parse_args()

    rows = benchmark_live() if args.live else benchmark_mock()
    write_report(rows, args.output)


if __name__ == "__main__":
    main()
