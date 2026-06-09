"""End-to-end RAG pipeline.

Two modes:
  - Build mode (scripts/ingest.py): load docs -> chunk -> embed -> save to disk
  - Query mode (scripts/query.py): load from disk -> retrieve -> generate
"""

from pathlib import Path
import pickle

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document

from src.config import settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import get_embedder
from src.ingestion.loader import load_documents
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.reranker import Reranker
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.dedup import deduplicate
from src.generation.llm import AnswerGenerator


BM25_PATH = Path(settings.index_dir) / "bm25.pkl"
CHROMA_PATH = Path(settings.index_dir) / "chroma"


def ingest(documents_dir=None):
    """Load documents, chunk, embed, and save index to disk."""
    source = documents_dir or settings.documents_dir
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print("Loading documents...")
    raw_docs = load_documents(source)

    print("Chunking...")
    chunks: list[Document] = chunk_documents(raw_docs)
    print(f"  {len(chunks)} chunks from {len(raw_docs)} documents")

    embedder = get_embedder()

    print("Building and saving vector store...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=str(CHROMA_PATH),
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("Building and saving BM25 index...")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = settings.retrieval_top_k
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"Done. Index saved to {index_dir}/")


class RAGPipeline:
    """Loads saved index from disk and answers queries."""

    def __init__(self):
        if not CHROMA_PATH.exists() or not BM25_PATH.exists():
            raise FileNotFoundError(
                "No index found. Run `python scripts/ingest.py` first."
            )

        print("Loading index from disk...")
        embedder = get_embedder()

        vectorstore = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=embedder,
        )
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_top_k},
        )

        with open(BM25_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[settings.vector_weight, settings.bm25_weight],
        )
        self.embedder = embedder
        self.expander = QueryExpander()
        self.reranker = Reranker()
        self.generator = AnswerGenerator()
        print("Ready.\n")

    def query(self, question: str) -> dict:
        # 1 — query expansion + HyDE
        queries = self.expander.expand(question)
        print(f"  Expanded to {len(queries)} queries")

        # 2 — ensemble retrieval per query
        all_docs: dict[str, Document] = {}
        per_query_ranked: list[list[str]] = []

        for q in queries:
            ranked_keys = []
            for doc in self.hybrid_retriever.invoke(q):
                key = doc.metadata.get("source", "") + doc.page_content[:60]
                if key not in all_docs:
                    all_docs[key] = doc
                ranked_keys.append(key)
            per_query_ranked.append(ranked_keys)

        # 3 — cross-query RRF fusion
        fused = reciprocal_rank_fusion(per_query_ranked, k=settings.rrf_k)
        candidates = [all_docs[cid] for cid, _ in fused if cid in all_docs]
        print(f"  {len(candidates)} candidates after fusion")

        # 4 — wrap as DocumentChunk for dedup + reranker
        from src.models import DocumentChunk
        chunk_objs = [
            DocumentChunk(
                chunk_id=str(i),
                text=doc.page_content,
                source=doc.metadata.get("source", ""),
                parent_id="",
                metadata=doc.metadata,
            )
            for i, doc in enumerate(candidates)
        ]

        # 5 — dedup
        chunk_objs = deduplicate(chunk_objs, self.embedder)

        # 6 — rerank
        reranked = self.reranker.rerank(question, chunk_objs)
        print(f"  Reranked to top {len(reranked)} chunks")

        # 7 — generate
        print("Generating answer...")
        answer = self.generator.generate(question, reranked)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "index": i,
                    "source": c.source.split("/")[-1],
                    "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
                }
                for i, c in enumerate(reranked, start=1)
            ],
        }

    def clear_history(self) -> None:
        self.generator.clear_history()
