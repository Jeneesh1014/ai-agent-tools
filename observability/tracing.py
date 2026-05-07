# observability/tracing.py

import os
import uuid
from typing import Dict, Any

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class TracingClient:
    def __init__(self):
        self.client = None
        # trace objects are not JSON-serializable so they can't live in AgentState
        # store them here keyed by trace_id — nodes look them up via state["trace_id"]
        # In newer Langfuse SDK versions (OTel-based), traces/spans are managed via
        # `start_as_current_observation(...)` context managers.
        self._active_traces: Dict[str, Any] = {}
        self._setup()

    def _setup(self):
        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                host=settings.LANGFUSE_HOST,
            )
            # auth_check is the correct way to verify connection in newer SDK versions
            # .trace() was removed as an auth mechanism
            self.client.auth_check()
            logger.info("Langfuse tracing connected")
        except Exception as e:
            # tracing must never crash the agent — degrade silently
            logger.warning(f"Langfuse setup failed, tracing disabled: {e}")
            self.client = None

    def generate_trace_id(self) -> str:
        return str(uuid.uuid4())

    def start_trace(self, trace_id: str, question: str) -> None:
        if not self.client:
            return
        try:
            # Enter a root observation for the whole agent run.
            root_cm = self.client.start_as_current_observation(
                as_type="span",
                name="agent-query",
                trace_context={"trace_id": trace_id},
                input={"question": question},
            )
            # Manually enter context so subsequent node spans become children
            # of this root observation in the same thread.
            root_cm.__enter__()
            self._active_traces[trace_id] = root_cm
            logger.info(f"Trace started: {trace_id[:8]}...")
        except Exception as e:
            logger.warning(f"Could not start trace: {e}")

    def log_router(self, trace_id: str, question: str, route: str) -> None:
        try:
            with self.client.start_as_current_observation(
                as_type="span",
                name="router_node",
                trace_context={"trace_id": trace_id},
                input={"question": question},
                output={"route": route},
            ):
                pass
        except Exception as e:
            logger.warning(f"Could not log router span: {e}")

    def log_rag(self, trace_id: str, result_count: int) -> None:
        try:
            with self.client.start_as_current_observation(
                as_type="span",
                name="rag_node",
                trace_context={"trace_id": trace_id},
                output={"result_count": result_count},
            ):
                pass
        except Exception as e:
            logger.warning(f"Could not log rag span: {e}")

    def log_web(self, trace_id: str, result_count: int) -> None:
        try:
            with self.client.start_as_current_observation(
                as_type="span",
                name="web_node",
                trace_context={"trace_id": trace_id},
                output={"result_count": result_count},
            ):
                pass
        except Exception as e:
            logger.warning(f"Could not log web span: {e}")

    def log_combine(self, trace_id: str, context_length: int) -> None:
        try:
            with self.client.start_as_current_observation(
                as_type="span",
                name="combine_node",
                trace_context={"trace_id": trace_id},
                output={"context_length": context_length},
            ):
                pass
        except Exception as e:
            logger.warning(f"Could not log combine span: {e}")

    def log_answer(self, trace_id: str, answer_length: int, model: str) -> None:
        try:
            with self.client.start_as_current_observation(
                as_type="span",
                name="answer_node",
                trace_context={"trace_id": trace_id},
                output={"answer_length": answer_length, "model": model},
            ):
                pass
        except Exception as e:
            logger.warning(f"Could not log answer span: {e}")

    def end_trace(self, trace_id: str) -> None:
        if not self.client:
            return
        try:
            root_cm = self._active_traces.pop(trace_id, None)
            if root_cm is not None:
                # Exit root context manager so the trace is properly closed.
                root_cm.__exit__(None, None, None)

            # Without flush, async spans may not get sent before the response returns.
            self.client.flush()
            logger.info(f"Trace flushed and closed: {trace_id[:8]}...")
        except Exception as e:
            logger.warning(f"Langfuse flush failed: {e}")