# config/settings.py

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = BASE_DIR / "data" / "documents"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
LOGS_PATH = BASE_DIR / "logs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
COLLECTION_NAME = "research_papers"
TOP_K = 8
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
RERANK_TOP_N = 3
COHERE_MODEL = "rerank-english-v3.0"

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS = 1024
TEMPERATURE = 0.1

# 5 iterations is plenty — if the agent hasn't finished by then something is wrong
MAX_ITERATIONS = 5
ROUTING_OPTIONS = ["rag", "web", "both"]

TAVILY_MAX_RESULTS = 5

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")