# agent/memory.py

from typing import List
from utils.logger import get_logger

logger = get_logger(__name__)


def format_history(chat_history: List) -> str:
    if not chat_history:
        return ""

    # cap at last 5 exchanges — full history blows the token budget fast
    recent = chat_history[-10:]

    lines = []
    for item in recent:
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    return "\n".join(lines)


def clear_history() -> List:
    logger.info("Conversation history cleared")
    return []