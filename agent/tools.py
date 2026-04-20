import os
from dataclasses import dataclass
from typing import List

from tavily import TavilyClient

from config import settings
from core.retrieval import Retriever
from core.reranking import Reranker
from entity.config_entity import RetrieverConfig, RerankerConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResult:
    content: str
    source: str
    relevance_score: float


@dataclass
class WebResult:
    content: str
    url: str
    title: str
    score: float


class RAGTool:
    def __init__(self):
        retriever_config = RetrieverConfig(
            documents_path=settings.DOCUMENTS_PATH,
            chroma_db_path=settings.CHROMA_DB_PATH,
            collection_name=settings.COLLECTION_NAME,
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_device=settings.EMBEDDING_DEVICE,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            min_chunk_length=settings.MIN_CHUNK_LENGTH,
            top_k=settings.TOP_K,
            vector_weight=settings.VECTOR_WEIGHT,
            bm25_weight=settings.BM25_WEIGHT,
        )

        reranker_config = RerankerConfig(
            cohere_api_key=os.environ["COHERE_API_KEY"],
            model=settings.COHERE_MODEL,
            top_n=settings.RERANK_TOP_N,
        )

        self.retriever = Retriever(retriever_config)
        self.reranker = Reranker(reranker_config)

        logger.info("Building RAG index — this takes a minute on first run")
        artifact = self.retriever.setup()
        logger.info(
            f"RAG index ready — {artifact.documents_processed} docs, "
            f"{artifact.chunks_created} chunks in {artifact.ingestion_time_seconds:.1f}s"
        )

    def search(self, query: str) -> List[RAGResult]:
        raw = self.retriever.retrieve(query)
        reranked = self.reranker.rerank(query, raw)

        results = [
            RAGResult(
                content=r["content"],
                source=r["source"],
                relevance_score=r["relevance_score"],
            )
            for r in reranked
        ]

        logger.info(f"RAG search returned {len(results)} results")
        return results


class WebSearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        logger.info("WebSearchTool ready")

    def search(self, query: str) -> List[WebResult]:
        logger.info(f"Web search: {query[:80]}...")

        response = self.client.search(
            query=query,
            max_results=settings.TAVILY_MAX_RESULTS,
            search_depth="basic",
        )

        results = [
            WebResult(
                content=r.get("content", ""),
                url=r.get("url", ""),
                title=r.get("title", ""),
                score=r.get("score", 0.0),
            )
            for r in response.get("results", [])
        ]

        logger.info(f"Web search returned {len(results)} results")
        return results