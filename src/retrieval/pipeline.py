"""Full retrieval pipeline orchestrator.

Stage flow:
  1. Multi-Query Expansion + HyDE
  2+3. EnsembleRetriever (Chroma vector + BM25, weighted RRF) — per query
  4. Cross-Query RRF Fusion
  5. Near-duplicate removal
  6. Cohere Reranking
"""

from src.config import settings
from src.ingestion.embedder import get_embedder
from src.ingestion.indexer import DualIndexStore
from src.models import DocumentChunk
from src.retrieval.dedup import deduplicate
from src.retrieval.ensemble import build_ensemble_retriever, retrieve_for_query
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.reranker import Reranker
from src.retrieval.rrf import reciprocal_rank_fusion, to_chunk_list


class RetrievalPipeline:
    def __init__(
        self,
        store: DualIndexStore,
        expander: QueryExpander | None = None,
        reranker: Reranker | None = None,
    ):
        self.store = store
        self.expander = expander or QueryExpander()
        self.reranker = reranker or Reranker()
        self.embedder = get_embedder()

        self.hybrid_retriever = build_ensemble_retriever(store)

        # Child chunk lookups
        self._chunk_lookup: dict[str, DocumentChunk] = {
            c.chunk_id: c for c in store.chunks
        }
        self._content_to_chunk_id: dict[str, str] = {
            c.text: c.chunk_id for c in store.chunks
        }

    def retrieve(self, query: str) -> list[DocumentChunk]:
        """Run all retrieval stages and return reranked chunks."""

        # Stage 1 — Multi-Query Expansion + HyDE
        queries = self.expander.expand(query)
        print(f"  Expanded to {len(queries)} queries (incl. HyDE)")

        # Stages 2+3 — EnsembleRetriever per query
        per_query_ranked: list[list[str]] = []
        for q in queries:
            ranked_ids = retrieve_for_query(q, self.hybrid_retriever, self._content_to_chunk_id)
            per_query_ranked.append(ranked_ids)

        # Stage 4 — Cross-Query RRF Fusion
        cross_fused = reciprocal_rank_fusion(per_query_ranked, k=settings.rrf_k)
        candidate_ids = [cid for cid, _ in cross_fused]
        candidates = to_chunk_list(
            [(cid, 0.0) for cid in candidate_ids],
            self._chunk_lookup,
        )
        print(f"  Cross-query fusion: {len(candidates)} candidates")

        # Stage 5 — Near-duplicate removal
        candidates = deduplicate(candidates, self.embedder)
        print(f"  After dedup: {len(candidates)} chunks")

        # Stage 6 — Cohere Reranking
        reranked = self.reranker.rerank(query, candidates)
        print(f"  Reranked to top {len(reranked)} chunks")

        return reranked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_source_boost(
    query: str,
    candidates: list[DocumentChunk],
) -> list[DocumentChunk]:
    """Promote chunks whose source filename tokens appear in the query."""
    q_lower = query.lower()
    source_stems: dict[str, list[str]] = {}
    for c in candidates:
        stem = c.source.split("/")[-1]
        source_stems.setdefault(stem, []).append(c.chunk_id)

    matched_stems = {
        stem
        for stem in source_stems
        for token in stem.split("_")
        if len(token) >= settings.company_boost_min_token_len and token in q_lower
    }

    if not matched_stems:
        return candidates

    matched = [c for c in candidates if c.source.split("/")[-1] in matched_stems]
    others = [c for c in candidates if c.source.split("/")[-1] not in matched_stems]
    return matched + others
