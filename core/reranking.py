import time
from typing import List, Dict

import cohere

from entity.config_entity import RerankerConfig
from entity.artifact_entity import RerankedArtifact
from utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.client = cohere.ClientV2(api_key=config.cohere_api_key)

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        if not results:
            return []

        start = time.time()
        documents = [r["content"] for r in results]

        response = self.client.rerank(
            model=self.config.model,
            query=query,
            documents=documents,
            top_n=self.config.top_n,
        )

        reranked = []
        for hit in response.results:
            original = results[hit.index]
            reranked.append({
                "content": original["content"],
                "source": original["source"],
                "relevance_score": hit.relevance_score,
            })

        elapsed = time.time() - start
        logger.info(f"Reranked {len(results)} → {len(reranked)} results in {elapsed:.2f}s")
        return reranked

    def initiate_reranking(self, query: str, results: List[Dict]) -> RerankedArtifact:
        start = time.time()
        reranked = self.rerank(query, results)
        elapsed = time.time() - start

        return RerankedArtifact(
            query=query,
            results_before=len(results),
            results_after=len(reranked),
            rerank_time_seconds=elapsed,
        )