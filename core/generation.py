import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from entity.config_entity import GeneratorConfig
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using only the context provided.
If the context does not contain enough information to answer, say so clearly.
Be concise, accurate, and cite sources when they are available in the context."""


class Generator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.llm = ChatGroq(
            api_key=config.groq_api_key,
            model=config.model,
            max_tokens=config.max_tokens,
            # low temp keeps answers grounded in the context, not creative
            temperature=config.temperature,
        )

    def generate(self, question: str, context: str) -> str:
        logger.info(f"Generating answer for: {question[:80]}...")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]

        response = self.llm.invoke(messages)
        answer = response.content.strip()

        logger.info(f"Generated {len(answer)} chars")
        return answer