from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.tools import RAGResult, RAGTool, WebResult, WebSearchTool


def make_ingestion_artifact():
    return SimpleNamespace(
        documents_processed=2,
        chunks_created=4,
        ingestion_time_seconds=0.1,
    )


def test_rag_tool_builds_retriever_and_reranker_once():
    with patch.dict("os.environ", {"COHERE_API_KEY": "test-cohere-key"}):
        with patch("agent.tools.Retriever") as retriever_cls:
            with patch("agent.tools.Reranker") as reranker_cls:
                retriever = MagicMock()
                retriever.setup.return_value = make_ingestion_artifact()
                retriever_cls.return_value = retriever

                tool = RAGTool()

    assert tool.retriever is retriever
    assert tool.reranker is reranker_cls.return_value
    retriever.setup.assert_called_once()
    retriever_cls.assert_called_once()
    reranker_cls.assert_called_once()


def test_rag_tool_search_returns_rag_results_in_correct_format():
    raw_results = [
        {"content": "Transformer content", "source": "transformer_paper.pdf", "score": 0.91},
        {"content": "BERT content", "source": "bert_paper.pdf", "score": 0.82},
    ]
    reranked_results = [
        {
            "content": "Transformer content",
            "source": "transformer_paper.pdf",
            "relevance_score": 0.98,
        },
        {
            "content": "BERT content",
            "source": "bert_paper.pdf",
            "relevance_score": 0.87,
        },
    ]

    with patch.dict("os.environ", {"COHERE_API_KEY": "test-cohere-key"}):
        with patch("agent.tools.Retriever") as retriever_cls:
            with patch("agent.tools.Reranker") as reranker_cls:
                retriever = MagicMock()
                retriever.setup.return_value = make_ingestion_artifact()
                retriever.retrieve.return_value = raw_results
                retriever_cls.return_value = retriever

                reranker = MagicMock()
                reranker.rerank.return_value = reranked_results
                reranker_cls.return_value = reranker

                tool = RAGTool()
                results = tool.search("What is self-attention?")

    assert results == [
        RAGResult(
            content="Transformer content",
            source="transformer_paper.pdf",
            relevance_score=0.98,
        ),
        RAGResult(
            content="BERT content",
            source="bert_paper.pdf",
            relevance_score=0.87,
        ),
    ]
    retriever.retrieve.assert_called_once_with("What is self-attention?")
    reranker.rerank.assert_called_once_with("What is self-attention?", raw_results)


def test_rag_tool_search_handles_empty_results():
    with patch.dict("os.environ", {"COHERE_API_KEY": "test-cohere-key"}):
        with patch("agent.tools.Retriever") as retriever_cls:
            with patch("agent.tools.Reranker") as reranker_cls:
                retriever = MagicMock()
                retriever.setup.return_value = make_ingestion_artifact()
                retriever.retrieve.return_value = []
                retriever_cls.return_value = retriever

                reranker = MagicMock()
                reranker.rerank.return_value = []
                reranker_cls.return_value = reranker

                tool = RAGTool()
                results = tool.search("Unknown topic")

    assert results == []
    reranker.rerank.assert_called_once_with("Unknown topic", [])


def test_web_search_tool_builds_tavily_client_with_api_key():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "test-tavily-key"}):
        with patch("agent.tools.TavilyClient") as tavily_cls:
            tool = WebSearchTool()

    tavily_cls.assert_called_once_with(api_key="test-tavily-key")
    assert tool.client is tavily_cls.return_value


def test_web_search_tool_returns_web_results_in_correct_format():
    tavily_response = {
        "results": [
            {
                "content": "Google released a new model.",
                "url": "https://example.com/google-model",
                "title": "Google model release",
                "score": 0.93,
            },
            {
                "content": "Another source about the release.",
                "url": "https://example.com/second",
                "title": "Second source",
                "score": 0.81,
            },
        ]
    }

    with patch.dict("os.environ", {"TAVILY_API_KEY": "test-tavily-key"}):
        with patch("agent.tools.TavilyClient") as tavily_cls:
            client = MagicMock()
            client.search.return_value = tavily_response
            tavily_cls.return_value = client

            tool = WebSearchTool()
            results = tool.search("What did Google release last week?")

    assert results == [
        WebResult(
            content="Google released a new model.",
            url="https://example.com/google-model",
            title="Google model release",
            score=0.93,
        ),
        WebResult(
            content="Another source about the release.",
            url="https://example.com/second",
            title="Second source",
            score=0.81,
        ),
    ]
    client.search.assert_called_once_with(
        query="What did Google release last week?",
        max_results=5,
        search_depth="basic",
    )


def test_web_search_tool_handles_empty_results():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "test-tavily-key"}):
        with patch("agent.tools.TavilyClient") as tavily_cls:
            client = MagicMock()
            client.search.return_value = {"results": []}
            tavily_cls.return_value = client

            tool = WebSearchTool()
            results = tool.search("No useful results")

    assert results == []


def test_web_search_tool_handles_missing_results_key():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "test-tavily-key"}):
        with patch("agent.tools.TavilyClient") as tavily_cls:
            client = MagicMock()
            client.search.return_value = {}
            tavily_cls.return_value = client

            tool = WebSearchTool()
            results = tool.search("No results key")

    assert results == []


def test_web_search_tool_handles_missing_optional_fields():
    with patch.dict("os.environ", {"TAVILY_API_KEY": "test-tavily-key"}):
        with patch("agent.tools.TavilyClient") as tavily_cls:
            client = MagicMock()
            client.search.return_value = {"results": [{"content": "Only content"}]}
            tavily_cls.return_value = client

            tool = WebSearchTool()
            results = tool.search("Sparse result")

    assert results == [
        WebResult(
            content="Only content",
            url="",
            title="",
            score=0.0,
        )
    ]
