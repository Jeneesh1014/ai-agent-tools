# Intelligent Research Agent

An AI research assistant that decides whether a question should be answered from local research papers, live web search, or both.

## Live Demo

🔗 https://your-app-name.onrender.com

## API Docs

🔗 https://your-app-name.onrender.com/docs

The core idea is routing. The agent does not blindly run every tool. It uses a LangGraph router node to choose one of three paths:

- `rag`: use local PDFs through hybrid retrieval and Cohere reranking
- `web`: use Tavily for current information
- `both`: combine internal research context with current web results

This is Project 2 in my AI Engineer portfolio, built after a complete RAG document assistant.

## What It Does

Example routing behavior:

| Question                                               | Route | Why                                            |
| ------------------------------------------------------ | ----- | ---------------------------------------------- |
| What is the attention mechanism?                       | RAG   | Foundational concept in the research documents |
| What LLM did Google release last week?                 | WEB   | Needs current information                      |
| How does LoRA compare with recent fine-tuning methods? | BOTH  | Needs paper context and recent work            |

The final answer includes cited sources, route metadata, result counts, and request latency.

## Architecture

```text
User Query
    |
    v
AgentState
    |
    v
Router Node
    |
    +---- rag ----> RAG Node ----+
    |                            |
    +---- web ----> Web Node ----+--> Combine Node --> Answer Node --> Response
    |                            |
    +---- both ---> RAG + Web ---+

Langfuse traces every agent run and node span.
```

## Tech Stack

| Layer               | Tool                                     |
| ------------------- | ---------------------------------------- |
| Agent orchestration | LangGraph                                |
| LLM                 | Groq `llama-3.1-8b-instant`              |
| Embeddings          | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB           | ChromaDB                                 |
| Hybrid search       | Vector search + BM25                     |
| Reranking           | Cohere `rerank-english-v3.0`             |
| Web search          | Tavily                                   |
| Observability       | Langfuse                                 |
| API                 | FastAPI                                  |
| UI                  | Gradio                                   |
| Tests               | pytest                                   |
| Lint                | Ruff                                     |
| Container           | Docker + Docker Compose                  |
| CI                  | GitHub Actions                           |

## Project Structure

```text
agent/               LangGraph state, graph, nodes, tools, memory
core/                Retrieval, reranking, generation
observability/       Langfuse tracing helpers
app/                 FastAPI API, Gradio UI, app runner
config/              Central settings
entity/              Dataclass configs/artifacts and Pydantic response models
tests/               Mocked routing, tool, graph, and API tests
scripts/             Evaluation and benchmark scripts
reports/             Generated evaluation and benchmark JSON reports
data/documents/      Research PDFs
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Required environment variables:

```bash
GROQ_API_KEY=...
COHERE_API_KEY=...
TAVILY_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
GROQ_MODEL=llama-3.1-8b-instant
```

## Run Locally

```bash
python app/run.py
```

Open:

```text
http://localhost:8000
```

API docs:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the attention mechanism?", "session_id": "demo"}'
```

## Docker

Build and run:

```bash
docker compose up --build
```

The app runs on:

```text
http://localhost:8000
```

Persistent volumes:

- `./chroma_db:/app/chroma_db`
- `./data:/app/data`
- `./logs:/app/logs`

## Tests

All external APIs are mocked in automated tests.

```bash
pytest tests/ -q
```

Current result:

```text
41 passed
```

Lint:

```bash
ruff check .
```

Smoke test:

```bash
python -c "from agent.graph import build_graph; print('ok')"
```

## Evaluation

The evaluation script checks whether the router chooses the expected tool for 10 representative questions.

Quota-free baseline mode:

```bash
python scripts/evaluate.py
```

Live agent mode:

```bash
python scripts/evaluate.py --live
```

Output:

```text
reports/evaluation_results.json
```

Current committed baseline:

| Metric           | Value                 |
| ---------------- | --------------------- |
| Cases            | 10                    |
| Correct          | 10                    |
| Routing accuracy | 100%                  |
| Mode             | Mock routing baseline |

The committed baseline is intentionally API-free. Use `--live` to regenerate with real Groq, Cohere, Tavily, and Langfuse calls.

## Benchmark

The benchmark script measures latency over 20 mixed questions.

Quota-free baseline mode:

```bash
python scripts/benchmark.py
```

Live agent mode:

```bash
python scripts/benchmark.py --live
```

Output:

```text
reports/benchmark_results.json
```

The mock benchmark is useful for validating the reporting pipeline. The live benchmark is the real portfolio number.

## API Response

`POST /ask` returns:

```json
{
  "question": "What is the attention mechanism?",
  "answer": "...",
  "route_taken": "rag",
  "sources": [
    {
      "title": "transformer_paper.pdf",
      "url_or_filename": "transformer_paper.pdf",
      "source_type": "rag",
      "relevance_score": 0.95
    }
  ],
  "model_used": "llama-3.1-8b-instant",
  "total_time_seconds": 1.23,
  "rag_results_count": 3,
  "web_results_count": 0
}
```

## CI

GitHub Actions runs:

- dependency installation
- mocked pytest suite
- Ruff lint
- graph import smoke test
- Docker image build test

Workflow:

```text
.github/workflows/ci.yml
```

## Design Decisions

- LangGraph is used directly instead of LangChain agents so the graph is explicit and testable.
- The graph is built once at API startup, not per request.
- RAG and web search are separate tools so the router decision is visible and measurable.
- Conversation memory keeps only the recent history to avoid token growth.
- Langfuse tracing is optional at runtime and never crashes the agent if unavailable.
- Tests mock Groq, Cohere, Tavily, and Langfuse to protect free-tier quota and keep CI deterministic.

## Portfolio Value

This project demonstrates:

- AI agent routing with LangGraph
- tool use across RAG and web search
- production-style tracing with Langfuse
- FastAPI and Gradio integration
- Dockerized deployment path
- CI with mocked external services
- evaluation and benchmark artifacts
