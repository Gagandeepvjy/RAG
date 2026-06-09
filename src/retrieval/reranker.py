"""Stage 5: Cohere reranking."""

import cohere
from langchain_core.documents import Document

from src.config import settings
from src.models import DocumentChunk


class Reranker:
    """Reranks candidate chunks using Cohere's rerank API."""

    def __init__(self):
        self._client = cohere.Client(api_key=settings.cohere_api_key)
        self._model = settings.cohere_rerank_model

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int | None = None,
    ) -> list[DocumentChunk]:
        """Return top-k chunks reordered by Cohere relevance score."""
        if not chunks:
            return []

        k = top_k or settings.rerank_top_k

        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=[c.text for c in chunks],
            top_n=k,
        )

        # response.results is ordered by relevance (best first)
        return [chunks[r.index] for r in response.results]
