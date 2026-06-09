"""End-to-end RAG pipeline — notebook style.

No persistence. On every startup:
  1. Load documents from disk
  2. Chunk them
  3. Build Chroma (in-memory) + BM25 retriever
  4. Wrap in EnsembleRetriever
Then answer queries against that.
"""

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


class RAGPipeline:
    """Loads documents, builds retrievers, answers queries. No disk index."""

    def __init__(self, documents_dir=None):
        source = documents_dir or settings.documents_dir

        print("Loading documents...")
        raw_docs = load_documents(source)

        print("Chunking...")
        chunks: list[Document] = chunk_documents(raw_docs)
        print(f"  {len(chunks)} chunks from {len(raw_docs)} documents")

        print("Building vector store...")
        embedder = get_embedder()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            collection_metadata={"hnsw:space": "cosine"},
        )
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_top_k},
        )

        print("Building BM25 index...")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = settings.retrieval_top_k

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[settings.vector_weight, settings.bm25_weight],
        )
        self.chunks = chunks
        self.embedder = embedder
        self.expander = QueryExpander()
        self.reranker = Reranker()
        self.generator = AnswerGenerator()
        print("Ready.\n")

    def query(self, question: str) -> dict:
        """Retrieve + generate."""

        # 1 — query expansion + HyDE
        queries = self.expander.expand(question)
        print(f"  Expanded to {len(queries)} queries")

        # 2 — ensemble retrieval per query
        # build a stable key per doc: source + first 60 chars of content
        all_docs: dict[str, Document] = {}   # key → doc (first seen wins)
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
