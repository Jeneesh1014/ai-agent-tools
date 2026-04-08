import time
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
import numpy as np

from entity.config_entity import RetrieverConfig
from entity.artifact_entity import IngestionArtifact, RetrievalArtifact
from utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.chunks: List[Dict] = []
        self.bm25 = None
        self.collection = None
        self.embedding_model = None

    def load_documents(self) -> List:
        pdf_files = list(self.config.documents_path.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {self.config.documents_path}")

        logger.info(f"Loading {len(pdf_files)} PDFs")
        docs = []
        for pdf in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf))
                pages = loader.load()
                docs.extend(pages)
            except Exception as e:
                logger.warning(f"Skipping {pdf.name} — {e}")

        logger.info(f"Loaded {len(docs)} pages from {len(pdf_files)} PDFs")
        return docs

    def split_chunks(self, docs: List) -> List[Dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        raw_chunks = splitter.split_documents(docs)

        # filter out noise — very short chunks are usually headers or page numbers
        chunks = []
        for chunk in raw_chunks:
            if len(chunk.page_content.strip()) >= self.config.min_chunk_length:
                chunks.append({
                    "content": chunk.page_content.strip(),
                    "source": Path(chunk.metadata.get("source", "unknown")).name,
                })

        logger.info(f"Split into {len(chunks)} chunks (filtered from {len(raw_chunks)})")
        return chunks

    def build_vector_store(self, chunks: List[Dict]) -> chromadb.Collection:
        logger.info("Loading embedding model")
        self.embedding_model = SentenceTransformer(
            self.config.embedding_model,
            device=self.config.embedding_device,
        )

        client = chromadb.PersistentClient(path=str(self.config.chroma_db_path))

        # delete and recreate so a re-run doesn't double the documents
        try:
            client.delete_collection(self.config.collection_name)
        except Exception:
            pass

        collection = client.create_collection(self.config.collection_name)

        texts = [c["content"] for c in chunks]
        sources = [c["source"] for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        logger.info(f"Embedding {len(texts)} chunks")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()

        # chroma has a batch limit — splitting avoids silent failures on large collections
        batch_size = 500
        for i in range(0, len(texts), batch_size):
            collection.add(
                ids=ids[i:i + batch_size],
                documents=texts[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                metadatas=[{"source": s} for s in sources[i:i + batch_size]],
            )

        logger.info(f"Vector store ready — {len(texts)} chunks in collection '{self.config.collection_name}'")
        return collection

    def build_bm25_index(self, chunks: List[Dict]) -> BM25Okapi:
        tokenized = [c["content"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index ready")
        return bm25

    def retrieve(self, query: str) -> List[Dict]:
        start = time.time()

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()[0]
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.config.top_k,
            include=["documents", "metadatas", "distances"],
        )

        vector_hits = {}
        for doc, meta, dist in zip(
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0],
        ):
            # chroma returns L2 distance — convert to a 0-1 similarity score
            score = 1 / (1 + dist)
            vector_hits[doc] = {"content": doc, "source": meta["source"], "vector_score": score}

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:self.config.top_k]

        bm25_hits = {}
        max_bm25 = bm25_scores[top_bm25_indices[0]] if bm25_scores[top_bm25_indices[0]] > 0 else 1
        for idx in top_bm25_indices:
            if bm25_scores[idx] > 0:
                chunk = self.chunks[idx]
                normalized = bm25_scores[idx] / max_bm25
                bm25_hits[chunk["content"]] = {
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "bm25_score": normalized,
                }

        all_contents = set(vector_hits.keys()) | set(bm25_hits.keys())
        merged = []
        for content in all_contents:
            v_score = vector_hits.get(content, {}).get("vector_score", 0.0)
            b_score = bm25_hits.get(content, {}).get("bm25_score", 0.0)
            combined = self.config.vector_weight * v_score + self.config.bm25_weight * b_score
            source = (
                vector_hits.get(content) or bm25_hits.get(content)
            )["source"]
            merged.append({"content": content, "source": source, "score": combined})

        merged.sort(key=lambda x: x["score"], reverse=True)
        results = merged[:self.config.top_k]

        elapsed = time.time() - start
        logger.info(f"Retrieved {len(results)} chunks for query in {elapsed:.2f}s")
        return results

    def initiate_retrieval(self, query: str) -> RetrievalArtifact:
        start = time.time()
        results = self.retrieve(query)
        elapsed = time.time() - start
        return RetrievalArtifact(
            query=query,
            chunks_retrieved=len(results),
            collection_name=self.config.collection_name,
            retrieval_time_seconds=elapsed,
        )

    def setup(self) -> IngestionArtifact:
        """Loads PDFs, chunks them, builds vector store and BM25. Call once at startup."""
        start = time.time()
        docs = self.load_documents()
        self.chunks = self.split_chunks(docs)
        self.collection = self.build_vector_store(self.chunks)
        self.bm25 = self.build_bm25_index(self.chunks)
        elapsed = time.time() - start

        return IngestionArtifact(
            documents_processed=len(list(self.config.documents_path.glob("*.pdf"))),
            chunks_created=len(self.chunks),
            collection_name=self.config.collection_name,
            ingestion_time_seconds=elapsed,
        )