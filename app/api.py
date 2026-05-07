from contextlib import asynccontextmanager
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.graph import create_agent, run_agent
from config import settings
from entity.agent_model import AgentResponse, Source
from utils.logger import get_logger

logger = get_logger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


def _build_sources(state: dict) -> List[Source]:
    rag_results = state.get("rag_results") or []
    web_results = state.get("web_results") or []

    sources: List[Source] = []

    for r in rag_results:
        filename = r.get("source", "")
        sources.append(
            Source(
                title=filename,
                url_or_filename=filename,
                source_type="rag",
                relevance_score=float(r.get("relevance_score", 0.0)),
            )
        )

    for r in web_results:
        url = r.get("url", "")
        title = r.get("title", "") or url
        sources.append(
            Source(
                title=title,
                url_or_filename=url,
                source_type="web",
                relevance_score=float(r.get("score", 0.0)),
            )
        )

    # De-dupe by (source_type, url_or_filename) while preserving order.
    seen = set()
    deduped: List[Source] = []
    for s in sources:
        key = (s.source_type, s.url_or_filename)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    return deduped


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API startup — building LangGraph agent once")
    graph, tracer = create_agent()

    # Keep the graph + tracer in app state so requests can share them.
    app.state.agent_graph = graph
    app.state.tracer = tracer
    app.state.sessions: Dict[str, list] = {}

    yield


def create_api_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan, title="Intelligent Research Agent")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": settings.GROQ_MODEL, "tools": ["rag", "web"]}

    @app.post("/clear")
    def clear(req: ClearRequest) -> dict:
        session_id = req.session_id
        sessions: Dict[str, list] = app.state.sessions
        sessions[session_id] = []
        logger.info(f"Session cleared: {session_id[:8]}...")
        return {"status": "cleared"}

    @app.post("/ask", response_model=AgentResponse)
    def ask(req: AskRequest) -> AgentResponse:
        graph = app.state.agent_graph
        tracer = app.state.tracer
        sessions: Dict[str, list] = app.state.sessions

        session_id = req.session_id or str(uuid.uuid4())
        chat_history = sessions.get(session_id, [])

        start = time.monotonic()
        final_state = run_agent(
            graph=graph,
            tracer=tracer,
            question=req.question,
            chat_history=chat_history,
        )
        elapsed = time.monotonic() - start

        updated_history = final_state.get("chat_history") or chat_history
        sessions[session_id] = updated_history

        rag_results = final_state.get("rag_results") or []
        web_results = final_state.get("web_results") or []

        sources = _build_sources(final_state)

        logger.info(
            f"/ask done — route={final_state.get('route')} "
            f"rag={len(rag_results)} web={len(web_results)} "
            f"time={elapsed:.2f}s"
        )

        return AgentResponse(
            question=req.question,
            answer=final_state.get("answer") or "",
            route_taken=final_state.get("route") or "rag",
            sources=sources,
            model_used=settings.GROQ_MODEL,
            total_time_seconds=float(elapsed),
            rag_results_count=len(rag_results),
            web_results_count=len(web_results),
        )

    return app


app = create_api_app()

