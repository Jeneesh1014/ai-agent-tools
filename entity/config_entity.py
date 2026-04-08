from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RetrieverConfig:
    documents_path: Path
    chroma_db_path: Path
    collection_name: str
    embedding_model: str
    embedding_device: str
    chunk_size: int
    chunk_overlap: int
    min_chunk_length: int
    top_k: int
    vector_weight: float
    bm25_weight: float


@dataclass
class RerankerConfig:
    cohere_api_key: str
    model: str
    top_n: int


@dataclass
class GeneratorConfig:
    groq_api_key: str
    model: str
    max_tokens: int
    temperature: float


@dataclass
class AgentConfig:
    retriever_config: RetrieverConfig
    reranker_config: RerankerConfig
    generator_config: GeneratorConfig
    tavily_api_key: str
    max_iterations: int
    routing_options: List[str]