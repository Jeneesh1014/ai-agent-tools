import uuid
from typing import Any, Dict, List, Tuple

import gradio as gr
import httpx

from utils.logger import get_logger

logger = get_logger(__name__)


def _route_label(route_taken: str) -> str:
    route_taken = (route_taken or "rag").lower().strip()
    if route_taken == "web":
        return "WEB"
    if route_taken == "both":
        return "BOTH"
    return "RAG"


def _format_sources(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "_No sources_"

    parts = []
    for s in sources:
        title = s.get("title", "") or s.get("url_or_filename", "")
        ref = s.get("url_or_filename", "")
        if ref and title != ref:
            parts.append(f"- {title} ({ref})")
        else:
            parts.append(f"- {ref or title}")
    return "\n".join(parts)


def _format_response(payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str, float]:
    answer = (payload.get("answer") or "").strip()
    route_taken = payload.get("route_taken") or "rag"
    total_time_seconds = float(payload.get("total_time_seconds") or 0.0)
    sources = payload.get("sources") or []

    assistant_md = (
        f"{answer}\n\n"
        f"**Route:** {_route_label(route_taken)}\n\n"
        f"**Sources used:**\n{_format_sources(sources)}\n\n"
        f"**Time:** {total_time_seconds:.2f}s"
    )

    return assistant_md, sources, route_taken, total_time_seconds


def build_ui(api_base_url: str = "http://localhost:8000") -> gr.Blocks:
    with gr.Blocks(title="Intelligent Research Agent") as demo:
        session_id_state = gr.State(str(uuid.uuid4()))

        chatbot = gr.Chatbot(label="Research Agent")

        message = gr.Textbox(
            label="Your question",
            placeholder="Ask something like: 'What is the attention mechanism?'",
            lines=2,
        )

        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear conversation")

        examples = [
            "What is the attention mechanism?",
            "What LLM did Google release last week?",
            "How does my research compare to the latest work?",
            "Explain BM25 and vector hybrid search tradeoffs.",
        ]

        gr.Examples(examples=examples, inputs=message, label="Example questions")

        def _ask(
            user_message: str,
            history: List[Dict[str, Any]],
            session_id: str,
        ) -> Tuple[List[Dict[str, Any]], str]:
            user_message = (user_message or "").strip()
            if not user_message:
                return history, session_id

            history = history or []
            # Gradio v6 expects message dicts: {"role": "user"/"assistant", "content": "..."}
            history = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "Thinking..."},
            ]

            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        f"{api_base_url}/ask",
                        json={"question": user_message, "session_id": session_id},
                    )
                    resp.raise_for_status()
                    payload = resp.json()

                assistant_md, _, _, _ = _format_response(payload)
                # replace the last assistant "Thinking..." entry
                history[-1] = {"role": "assistant", "content": assistant_md}

            except Exception as e:
                logger.error(f"UI /ask failed: {e}")
                history[-1] = {
                    "role": "assistant",
                    "content": "I couldn't reach the API. Check that the server is running and your API keys are set.",
                }

            return history, session_id

        def _clear(session_id: str) -> Tuple[List[Dict[str, Any]], str]:
            new_session_id = str(uuid.uuid4())
            try:
                with httpx.Client(timeout=30) as client:
                    client.post(f"{api_base_url}/clear", json={"session_id": session_id}).raise_for_status()
            except Exception as e:
                # clear locally even if API clear fails
                logger.warning(f"UI /clear failed (local reset anyway): {e}")

            return [], new_session_id

        send_btn.click(
            fn=_ask,
            inputs=[message, chatbot, session_id_state],
            outputs=[chatbot, session_id_state],
        )

        clear_btn.click(
            fn=_clear,
            inputs=[session_id_state],
            outputs=[chatbot, session_id_state],
        )

        # Convenience: press Enter to send.
        message.submit(
            fn=_ask,
            inputs=[message, chatbot, session_id_state],
            outputs=[chatbot, session_id_state],
        )

    return demo

