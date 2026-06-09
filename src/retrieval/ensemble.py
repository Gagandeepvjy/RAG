"""Stage 2+3: LangChain EnsembleRetriever (Chroma vector + BM25) with weighted RRF.

EnsembleRetriever returns child chunks. We also extract the parent_id from
metadata so the pipeline can later swap child chunks for their parent chunks.
"""

from langchain_core.documents import Document

from src.config import settings
from src.ingestion.indexer import DualIndexStore

try:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except ImportError:
    from langchain_community.retrievers.ensemble import EnsembleRetriever  # type: ignore[no-redef]


def build_ensemble_retriever(store: DualIndexStore) -> EnsembleRetriever:
    """
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3],
    )
    """
    vector_retriever = store.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )

    bm25_retriever = store.bm25_retriever
    bm25_retriever.k = settings.retrieval_top_k

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[settings.vector_weight, settings.bm25_weight],
    )


def retrieve_for_query(
    query: str,
    hybrid_retriever: EnsembleRetriever,
    content_to_chunk_id: dict[str, str],
) -> list[str]:
    """Run the hybrid retriever; return ranked child chunk_ids."""
    docs: list[Document] = hybrid_retriever.invoke(query)
    chunk_ids = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id") or content_to_chunk_id.get(doc.page_content)
        if cid:
            chunk_ids.append(cid)
    return chunk_ids
