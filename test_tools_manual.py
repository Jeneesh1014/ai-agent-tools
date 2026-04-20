"""
Manual smoke test for RAGTool and WebSearchTool.
Run from project root: python test_tools_manual.py

RAGTool first run embeds all PDFs — expect 2-4 minutes.
Subsequent runs are faster because ChromaDB is persisted.
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from utils.logger import get_logger

logger = get_logger("manual_test")


def test_web_search():
    logger.info("--- WebSearchTool test ---")
    from agent.tools import WebSearchTool

    tool = WebSearchTool()
    results = tool.search("latest LLM released by Google 2024")

    assert len(results) > 0, "Web search returned nothing"
    for r in results:
        print(f"\n  Title: {r.title}")
        print(f"  URL:   {r.url}")
        print(f"  Score: {r.score:.3f}")
        print(f"  Snippet: {r.content[:120]}...")

    logger.info(f"WebSearchTool OK — {len(results)} results")
    return True


def test_rag_search():
    logger.info("--- RAGTool test ---")
    from agent.tools import RAGTool

    tool = RAGTool()
    results = tool.search("What is the attention mechanism in transformers?")

    assert len(results) > 0, "RAG search returned nothing"
    for r in results:
        print(f"\n  Source:    {r.source}")
        print(f"  Relevance: {r.relevance_score:.3f}")
        print(f"  Content:   {r.content[:120]}...")

    logger.info(f"RAGTool OK — {len(results)} results")
    return True


if __name__ == "__main__":
    failures = []

    logger.info("Starting Day 3 manual tests")

    try:
        test_web_search()
        print("\nWebSearchTool PASS")
    except Exception as e:
        logger.error(f"WebSearchTool FAIL — {e}")
        failures.append("web_search")

    try:
        test_rag_search()
        print("\nRAGTool PASS")
    except Exception as e:
        logger.error(f"RAGTool FAIL — {e}")
        failures.append("rag_search")

    if failures:
        print(f"\nFailed: {failures}")
        sys.exit(1)
    else:
        print("\nAll Day 3 tests passed")