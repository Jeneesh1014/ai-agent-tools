from dataclasses import dataclass


@dataclass
class IngestionArtifact:
    documents_processed: int
    chunks_created: int
    collection_name: str
    ingestion_time_seconds: float


@dataclass
class RetrievalArtifact:
    query: str
    chunks_retrieved: int
    collection_name: str
    retrieval_time_seconds: float


@dataclass
class RerankedArtifact:
    query: str
    results_before: int
    results_after: int
    rerank_time_seconds: float