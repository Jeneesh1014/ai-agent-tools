from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api import _build_sources, create_api_app


def make_final_state(route="rag"):
    rag_results = []
    web_results = []

    if route in {"rag", "both"}:
        rag_results = [
            {
                "content": "Attention lets tokens compare information.",
                "source": "transformer_paper.pdf",
                "relevance_score": 0.95,
            }
        ]

    if route in {"web", "both"}:
        web_results = [
            {
                "content": "Recent model release details.",
                "url": "https://example.com/model",
                "title": "Model release",
                "score": 0.9,
            }
        ]

    return {
        "question": "Test question",
        "route": route,
        "rag_results": rag_results,
        "web_results": web_results,
        "combined_context": "Combined context",
        "answer": "Final mocked answer.",
        "sources": ["transformer_paper.pdf", "https://example.com/model"],
        "chat_history": [
            {"role": "user", "content": "Test question"},
            {"role": "assistant", "content": "Final mocked answer."},
        ],
        "trace_id": "test-trace-id",
    }


def make_client(final_state=None):
    graph = MagicMock(name="graph")
    tracer = MagicMock(name="tracer")

    create_agent_patch = patch("app.api.create_agent", return_value=(graph, tracer))
    run_agent_patch = patch(
        "app.api.run_agent",
        return_value=final_state or make_final_state(),
    )

    create_agent_mock = create_agent_patch.start()
    run_agent_mock = run_agent_patch.start()

    app = create_api_app()
    client = TestClient(app)

    return client, create_agent_mock, run_agent_mock, create_agent_patch, run_agent_patch


def close_client(client, *patches):
    client.close()
    for active_patch in patches:
        active_patch.stop()


def test_health_returns_status_model_and_tools():
    client, create_agent_mock, _, create_agent_patch, run_agent_patch = make_client()

    with client:
        response = client.get("/health")

    close_client(client, create_agent_patch, run_agent_patch)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "llama-3.1-8b-instant",
        "tools": ["rag", "web"],
    }
    create_agent_mock.assert_called_once()


def test_ask_returns_agent_response():
    client, _, run_agent_mock, create_agent_patch, run_agent_patch = make_client(
        make_final_state(route="both")
    )

    with client:
        response = client.post(
            "/ask",
            json={"question": "Compare attention with latest LLM work.", "session_id": "s1"},
        )

    close_client(client, create_agent_patch, run_agent_patch)

    body = response.json()
    assert response.status_code == 200
    assert body["question"] == "Compare attention with latest LLM work."
    assert body["answer"] == "Final mocked answer."
    assert body["route_taken"] == "both"
    assert body["model_used"] == "llama-3.1-8b-instant"
    assert body["rag_results_count"] == 1
    assert body["web_results_count"] == 1
    assert len(body["sources"]) == 2
    run_agent_mock.assert_called_once()


def test_ask_passes_shared_graph_and_tracer_to_agent_runner():
    client, _, run_agent_mock, create_agent_patch, run_agent_patch = make_client()

    with client:
        response = client.post(
            "/ask",
            json={"question": "What is RAG?", "session_id": "shared-deps"},
        )

    graph_arg = run_agent_mock.call_args.kwargs["graph"]
    tracer_arg = run_agent_mock.call_args.kwargs["tracer"]
    close_client(client, create_agent_patch, run_agent_patch)

    assert response.status_code == 200
    assert graph_arg._mock_name == "graph"
    assert tracer_arg._mock_name == "tracer"


def test_ask_reuses_session_history_on_follow_up():
    client, _, run_agent_mock, create_agent_patch, run_agent_patch = make_client()

    with client:
        first = client.post(
            "/ask",
            json={"question": "What is attention?", "session_id": "session-a"},
        )
        second = client.post(
            "/ask",
            json={"question": "Explain it more simply.", "session_id": "session-a"},
        )

    close_client(client, create_agent_patch, run_agent_patch)

    assert first.status_code == 200
    assert second.status_code == 200
    first_call = run_agent_mock.call_args_list[0].kwargs
    second_call = run_agent_mock.call_args_list[1].kwargs
    assert first_call["chat_history"] == []
    assert second_call["chat_history"] == [
        {"role": "user", "content": "Test question"},
        {"role": "assistant", "content": "Final mocked answer."},
    ]


def test_ask_generates_session_id_when_missing():
    client, _, run_agent_mock, create_agent_patch, run_agent_patch = make_client()

    with client:
        response = client.post("/ask", json={"question": "What is RAG?"})

    close_client(client, create_agent_patch, run_agent_patch)

    assert response.status_code == 200
    assert run_agent_mock.call_args.kwargs["question"] == "What is RAG?"


def test_clear_resets_session_history():
    client, _, run_agent_mock, create_agent_patch, run_agent_patch = make_client()

    with client:
        first = client.post(
            "/ask",
            json={"question": "What is attention?", "session_id": "session-clear"},
        )
        clear = client.post("/clear", json={"session_id": "session-clear"})
        second = client.post(
            "/ask",
            json={"question": "Start fresh.", "session_id": "session-clear"},
        )

    close_client(client, create_agent_patch, run_agent_patch)

    assert first.status_code == 200
    assert clear.status_code == 200
    assert clear.json() == {"status": "cleared"}
    assert second.status_code == 200
    second_call = run_agent_mock.call_args_list[1].kwargs
    assert second_call["chat_history"] == []


def test_ask_rejects_empty_question():
    client, _, _, create_agent_patch, run_agent_patch = make_client()

    with client:
        response = client.post("/ask", json={"question": ""})

    close_client(client, create_agent_patch, run_agent_patch)

    assert response.status_code == 422


def test_clear_rejects_missing_session_id():
    client, _, _, create_agent_patch, run_agent_patch = make_client()

    with client:
        response = client.post("/clear", json={})

    close_client(client, create_agent_patch, run_agent_patch)

    assert response.status_code == 422


def test_build_sources_deduplicates_rag_and_web_sources():
    state = {
        "rag_results": [
            {
                "source": "transformer_paper.pdf",
                "relevance_score": 0.95,
            },
            {
                "source": "transformer_paper.pdf",
                "relevance_score": 0.88,
            },
        ],
        "web_results": [
            {
                "url": "https://example.com/model",
                "title": "Model release",
                "score": 0.9,
            },
            {
                "url": "https://example.com/model",
                "title": "Duplicate model release",
                "score": 0.8,
            },
        ],
    }

    sources = _build_sources(state)

    assert len(sources) == 2
    assert sources[0].url_or_filename == "transformer_paper.pdf"
    assert sources[0].source_type == "rag"
    assert sources[1].url_or_filename == "https://example.com/model"
    assert sources[1].source_type == "web"


def test_build_sources_uses_url_as_title_when_web_title_missing():
    sources = _build_sources(
        {
            "rag_results": [],
            "web_results": [
                {
                    "url": "https://example.com/no-title",
                    "title": "",
                    "score": 0.7,
                }
            ],
        }
    )

    assert sources[0].title == "https://example.com/no-title"


def test_build_sources_returns_empty_list_when_state_has_no_results():
    assert _build_sources({}) == []
