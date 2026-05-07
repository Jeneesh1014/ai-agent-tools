from agent.graph import build_graph, run_agent
from entity.agent_model import AgentResponse, Source


def make_state(question: str, route=None, chat_history=None):
    return {
        "question": question,
        "route": route,
        "rag_results": None,
        "web_results": None,
        "combined_context": None,
        "answer": None,
        "sources": None,
        "chat_history": chat_history or [],
        "trace_id": "test-trace-id",
    }


class FakeNodes:
    def __init__(self, route: str):
        self.route = route
        self.calls = []

    def router_node(self, state):
        self.calls.append("router_node")
        return {"route": self.route}

    def rag_node(self, state):
        self.calls.append("rag_node")
        return {
            "rag_results": [
                {
                    "content": "Attention lets tokens exchange information.",
                    "source": "transformer_paper.pdf",
                    "relevance_score": 0.95,
                }
            ]
        }

    def web_node(self, state):
        self.calls.append("web_node")
        return {
            "web_results": [
                {
                    "content": "A recent model was released this week.",
                    "url": "https://example.com/latest-model",
                    "title": "Latest model release",
                    "score": 0.91,
                }
            ]
        }

    def combine_node(self, state):
        self.calls.append("combine_node")
        rag_results = state.get("rag_results") or []
        web_results = state.get("web_results") or []
        context = []
        sources = []

        for result in rag_results:
            context.append(result["content"])
            sources.append(result["source"])

        for result in web_results:
            context.append(result["content"])
            sources.append(result["url"])

        return {
            "combined_context": "\n".join(context),
            "sources": sources,
        }

    def answer_node(self, state):
        self.calls.append("answer_node")
        answer = f"Answered with {state.get('route')} route."
        return {
            "answer": answer,
            "chat_history": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": answer},
            ],
        }


class FakeTracer:
    def __init__(self):
        self.started = []
        self.ended = []

    def generate_trace_id(self):
        return "test-trace-id"

    def start_trace(self, trace_id, question):
        self.started.append((trace_id, question))

    def end_trace(self, trace_id):
        self.ended.append(trace_id)


def test_graph_runs_rag_route_end_to_end():
    nodes = FakeNodes(route="rag")
    graph = build_graph(nodes)

    final_state = graph.invoke(make_state("What is attention?"))

    assert final_state["route"] == "rag"
    assert len(final_state["rag_results"]) == 1
    assert final_state["web_results"] is None
    assert final_state["sources"] == ["transformer_paper.pdf"]
    assert final_state["answer"] == "Answered with rag route."
    assert "web_node" not in nodes.calls


def test_graph_runs_web_route_end_to_end():
    nodes = FakeNodes(route="web")
    graph = build_graph(nodes)

    final_state = graph.invoke(make_state("What model was released last week?"))

    assert final_state["route"] == "web"
    assert final_state["rag_results"] is None
    assert len(final_state["web_results"]) == 1
    assert final_state["sources"] == ["https://example.com/latest-model"]
    assert final_state["answer"] == "Answered with web route."
    assert "rag_node" not in nodes.calls


def test_graph_runs_both_route_and_merges_state():
    nodes = FakeNodes(route="both")
    graph = build_graph(nodes)

    final_state = graph.invoke(make_state("Compare transformers with the latest models."))

    assert final_state["route"] == "both"
    assert len(final_state["rag_results"]) == 1
    assert len(final_state["web_results"]) == 1
    assert final_state["sources"] == [
        "transformer_paper.pdf",
        "https://example.com/latest-model",
    ]
    assert final_state["answer"] == "Answered with both route."


def test_graph_defaults_unknown_route_to_rag_path():
    nodes = FakeNodes(route="unexpected")
    graph = build_graph(nodes)

    final_state = graph.invoke(make_state("Explain CNNs."))

    assert final_state["route"] == "unexpected"
    assert len(final_state["rag_results"]) == 1
    assert final_state["web_results"] is None
    assert "rag_node" in nodes.calls
    assert "web_node" not in nodes.calls


def test_run_agent_creates_trace_and_returns_final_state():
    nodes = FakeNodes(route="rag")
    graph = build_graph(nodes)
    tracer = FakeTracer()

    final_state = run_agent(
        graph=graph,
        tracer=tracer,
        question="Explain RAG.",
        chat_history=[{"role": "user", "content": "Previous question"}],
    )

    assert tracer.started == [("test-trace-id", "Explain RAG.")]
    assert tracer.ended == ["test-trace-id"]
    assert final_state["trace_id"] == "test-trace-id"
    assert final_state["answer"] == "Answered with rag route."


def test_graph_conversation_history_accumulates():
    nodes = FakeNodes(route="rag")
    graph = build_graph(nodes)
    initial_history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]

    final_state = graph.invoke(make_state("Follow-up question", chat_history=initial_history))

    assert final_state["chat_history"] == [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Follow-up question"},
        {"role": "assistant", "content": "Answered with rag route."},
    ]


def test_agent_response_model_accepts_graph_output():
    response = AgentResponse(
        question="What is attention?",
        answer="Attention lets tokens exchange information.",
        route_taken="rag",
        sources=[
            Source(
                title="transformer_paper.pdf",
                url_or_filename="transformer_paper.pdf",
                source_type="rag",
                relevance_score=0.95,
            )
        ],
        model_used="llama-3.1-8b-instant",
        total_time_seconds=0.42,
        rag_results_count=1,
        web_results_count=0,
    )

    assert response.route_taken == "rag"
    assert response.sources[0].source_type == "rag"
