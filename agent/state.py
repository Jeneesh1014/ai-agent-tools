from typing import TypedDict, List, Optional, Annotated
import operator


class AgentState(TypedDict):
    question: str
    route: Optional[str]
    rag_results: Optional[List]
    web_results: Optional[List]
    combined_context: Optional[str]
    answer: Optional[str]
    sources: Optional[List[str]]
    chat_history: Annotated[List, operator.add]
    trace_id: Optional[str]