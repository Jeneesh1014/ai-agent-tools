from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langgraph.types import Send

from agent.graph import route_decision
from agent.nodes import AgentNodes, ROUTING_PROMPT


def make_state(question: str, chat_history=None):
    return {
        "question": question,
        "route": None,
        "rag_results": None,
        "web_results": None,
        "combined_context": None,
        "answer": None,
        "sources": None,
        "chat_history": chat_history or [],
        "trace_id": "test-trace-id",
    }


def make_nodes(router_response: str):
    router_llm = MagicMock()
    router_llm.invoke.return_value = SimpleNamespace(content=router_response)
    answer_llm = MagicMock()

    with patch.dict("os.environ", {"GROQ_API_KEY": "test-groq-key"}):
        with patch("agent.nodes.ChatGroq", side_effect=[router_llm, answer_llm]):
            tracer = MagicMock()
            nodes = AgentNodes(
                rag_tool=MagicMock(),
                web_tool=MagicMock(),
                generator=MagicMock(),
                tracer=tracer,
            )

    return nodes, router_llm, answer_llm, tracer


@pytest.mark.parametrize(
    ("question", "llm_response", "expected_route"),
    [
        ("What is the attention mechanism in transformers?", "rag", "rag"),
        ("Explain BERT masked language modeling from the papers.", "RAG", "rag"),
        ("What LLM did Google release last week?", "web", "web"),
        ("What is the latest Groq model available today?", "WEB", "web"),
        (
            "Compare retrieval augmented generation with the latest agentic RAG work.",
            "both",
            "both",
        ),
        (
            "How does LoRA compare with recent parameter-efficient fine-tuning news?",
            "BOTH",
            "both",
        ),
    ],
)
def test_router_node_returns_expected_route(question, llm_response, expected_route):
    nodes, router_llm, _, tracer = make_nodes(llm_response)
    state = make_state(question)

    result = nodes.router_node(state)

    assert result == {"route": expected_route}
    router_llm.invoke.assert_called_once()
    tracer.log_router.assert_called_once_with(
        "test-trace-id",
        question,
        expected_route,
    )


def test_router_node_uses_chat_history_in_prompt():
    nodes, router_llm, _, _ = make_nodes("rag")
    state = make_state(
        "Can you explain it more simply?",
        chat_history=[
            {"role": "user", "content": "What is self-attention?"},
            {
                "role": "assistant",
                "content": "Self-attention lets tokens attend to each other.",
            },
        ],
    )

    nodes.router_node(state)

    messages = router_llm.invoke.call_args.args[0]
    prompt = messages[0].content
    assert "User: What is self-attention?" in prompt
    assert "Assistant: Self-attention lets tokens attend to each other." in prompt
    assert "Question: Can you explain it more simply?" in prompt


def test_router_node_uses_none_when_history_is_empty():
    nodes, router_llm, _, _ = make_nodes("web")

    nodes.router_node(make_state("What happened in AI this week?"))

    messages = router_llm.invoke.call_args.args[0]
    prompt = messages[0].content
    assert "Recent conversation:\nNone" in prompt


def test_router_node_prefers_both_when_response_contains_multiple_route_words():
    nodes, _, _, _ = make_nodes("Use both rag and web.")

    result = nodes.router_node(make_state("Compare transformer basics with latest work."))

    assert result["route"] == "both"


def test_router_node_defaults_to_rag_for_unrecognized_response():
    nodes, _, _, _ = make_nodes("internal documents")

    result = nodes.router_node(make_state("Explain convolutional neural networks."))

    assert result["route"] == "rag"


def test_router_prompt_contains_only_supported_route_options():
    assert "rag" in ROUTING_PROMPT
    assert "web" in ROUTING_PROMPT
    assert "both" in ROUTING_PROMPT
    assert "Reply with exactly one word" in ROUTING_PROMPT


@pytest.mark.parametrize(
    ("route", "expected_node"),
    [
        ("rag", "rag_node"),
        ("web", "web_node"),
    ],
)
def test_route_decision_sends_single_tool_routes(route, expected_node):
    state = make_state("test question")
    state["route"] = route

    decision = route_decision(state)

    assert isinstance(decision, Send)
    assert decision.node == expected_node
    assert decision.arg == state


def test_route_decision_sends_both_tools_in_parallel():
    state = make_state("Compare paper knowledge with recent AI news.")
    state["route"] = "both"

    decision = route_decision(state)

    assert [send.node for send in decision] == ["rag_node", "web_node"]
    assert [send.arg for send in decision] == [state, state]


def test_route_decision_defaults_to_rag_when_route_is_missing():
    state = make_state("Explain residual connections.")
    state["route"] = None

    decision = route_decision(state)

    assert isinstance(decision, Send)
    assert decision.node == "rag_node"
    assert decision.arg == state
