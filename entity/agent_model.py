from pydantic import BaseModel
from typing import List


class Source(BaseModel):
    title: str
    url_or_filename: str
    source_type: str
    relevance_score: float


class AgentResponse(BaseModel):
    question: str
    answer: str
    route_taken: str
    sources: List[Source]
    model_used: str
    total_time_seconds: float
    rag_results_count: int
    web_results_count: int