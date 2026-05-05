import os
from typing import Dict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from agent.memory import format_history

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

ANSWER_PROMPT = """You are a research assistant. Answer the question using the context provided.
If the context does not contain enough information, say so clearly.
Be concise and accurate. When citing sources, use the filename or URL provided in the context.

Context:
{context}

Question: {question}

Answer:"""


class AgentNodes:
    def __init__(self, rag_tool: RAGTool, web_tool: WebSearchTool, generator: Generator):
        self.rag_tool = rag_tool
        self.web_tool = web_tool
        self.generator = generator

        # max_tokens=10 because we only need one word back
        self.router_llm = ChatGroq(
            api_key=os.environ["GROQ_API_KEY"],
            model=settings.GROQ_MODEL,
            max_tokens=10,
            temperature=0.0,
        )

    def router_node(self, state: AgentState) -> Dict:
        question = state["question"]
        history = format_history(state.get("chat_history", []))

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
        rag_results = state.get("rag_results") or []
        web_results = state.get("web_results") or []
        route = state.get("route", "rag")

        context_parts = []
        sources = []

        if rag_results:
            context_parts.append("=== Research Documents ===")
            for r in rag_results:
                context_parts.append(f"[Source: {r['source']}]\n{r['content']}")
                sources.append(r["source"])

        if web_results:
            context_parts.append("=== Web Search Results ===")
            for r in web_results:
                context_parts.append(f"[Source: {r['title']} — {r['url']}]\n{r['content']}")
                sources.append(r["url"])

        combined_context = "\n\n".join(context_parts)

        # deduplicate while preserving order
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)

        logger.info(
            f"Combined context: {len(rag_results)} RAG + {len(web_results)} web results, "
            f"{len(combined_context)} chars"
        )
        return {"combined_context": combined_context, "sources": unique_sources}

    def answer_node(self, state: AgentState) -> Dict:
        question = state["question"]
        context = state.get("combined_context", "")
        route = state.get("route", "rag")

        if not context:
            logger.warning("answer_node received empty context")
            answer = "I was unable to find relevant information to answer your question."
        else:
            messages = [
                HumanMessage(
                    content=ANSWER_PROMPT.format(context=context, question=question)
                )
            ]
            # generator.generate expects context + question — build the prompt ourselves
            # so we can inject the full structured context cleanly
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                api_key=os.environ["GROQ_API_KEY"],
                model=settings.GROQ_MODEL,
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
            )
            response = llm.invoke(messages)
            answer = response.content.strip()

        logger.info(f"Answer generated: {len(answer)} chars via route '{route}'")

        # append to history so future turns have context
        new_history_entries = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        return {"answer": answer, "chat_history": new_history_entries}


