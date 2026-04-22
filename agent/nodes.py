import os
from typing import Dict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from agent.tools import RAGTool, WebSearchTool
from core.generation import Generator
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

ROUTING_PROMPT = """You are a routing agent deciding which tool to use to answer a research question.

Tools available:
- rag: Search internal research documents (PDFs and papers). Use for questions about concepts, theories, algorithms, architectures, methods, or anything that would appear in academic literature.
- web: Search the internet. Use for recent news, latest model releases, current events, or anything requiring up-to-date information not likely in research papers.
- both: Use both tools. Use when the question needs foundational knowledge from papers AND current/recent information together.

Recent conversation:
{history}

Question: {question}

Reply with exactly one word: rag, web, or both."""


class AgentNodes:
    def __init__(self, rag_tool: RAGTool, web_tool: WebSearchTool, generator: Generator):
        self.rag_tool = rag_tool
        self.web_tool = web_tool
        self.generator = generator

        # max_tokens=10 because we only need one word back — no point paying for more
        self.router_llm = ChatGroq(
            api_key=os.environ["GROQ_API_KEY"],
            model=settings.GROQ_MODEL,
            max_tokens=10,
            temperature=0.0,
        )

    def router_node(self, state: AgentState) -> Dict:
        question = state["question"]
        history = _format_history(state.get("chat_history", []))

        prompt = ROUTING_PROMPT.format(
            history=history if history else "None",
            question=question,
        )

        response = self.router_llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip().lower()

        # check "both" first — "rag" is a substring match risk if we go in the wrong order
        if "both" in raw:
            route = "both"
        elif "web" in raw:
            route = "web"
        else:
            route = "rag"

        logger.info(f"Router: '{question[:70]}' → {route}")
        return {"route": route}

    def rag_node(self, state: AgentState) -> Dict:
        results = self.rag_tool.search(state["question"])

        # convert dataclasses to plain dicts so LangGraph state stays serializable
        serialized = [
            {
                "content": r.content,
                "source": r.source,
                "relevance_score": r.relevance_score,
            }
            for r in results
        ]

        logger.info(f"RAG node: {len(serialized)} results")
        return {"rag_results": serialized}

    def web_node(self, state: AgentState) -> Dict:
        results = self.web_tool.search(state["question"])

        serialized = [
            {
                "content": r.content,
                "url": r.url,
                "title": r.title,
                "score": r.score,
            }
            for r in results
        ]

        logger.info(f"Web node: {len(serialized)} results")
        return {"web_results": serialized}

    def combine_node(self, state: AgentState) -> Dict:
        # stub — Day 5
        return {}

    def answer_node(self, state: AgentState) -> Dict:
        # stub — Day 5
        return {}


def _format_history(history: list) -> str:
    if not history:
        return ""

    # cap at last 5 exchanges — full history blows the token budget fast
    recent = history[-10:]
    lines = []
    for item in recent:
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    return "\n".join(lines)