"""Near-duplicate chunk removal based on cosine similarity.

After cross-query RRF fusion, multiple queries can surface chunks that are
nearly identical (e.g. two overlapping child chunks from the same passage).
Sending duplicates to the reranker wastes its token budget and can bias the
LLM toward repeated content. This module removes them greedily:

  - Walk the ranked list from best to worst.
  - Keep a chunk only if its cosine similarity to every already-kept chunk
    is below the configured threshold.
"""

import math

from src.config import settings
from src.models import DocumentChunk


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    denom = _norm(a) * _norm(b)
    return _dot(a, b) / denom if denom else 0.0


def deduplicate(
    chunks: list[DocumentChunk],
    embedder,
    threshold: float | None = None,
) -> list[DocumentChunk]:
    """
    Remove near-duplicate chunks. Preserves the original ranking order among
    kept chunks. Returns at most len(chunks) items (usually fewer).
    """
    if len(chunks) <= 1:
        return chunks

    t = threshold if threshold is not None else settings.dedup_threshold

    # Embed all chunk texts in one batch for efficiency
    texts = [c.text for c in chunks]
    embeddings: list[list[float]] = embedder.embed_documents(texts)

    kept_chunks: list[DocumentChunk] = []
    kept_embeddings: list[list[float]] = []

    for chunk, emb in zip(chunks, embeddings):
        is_duplicate = any(
            cosine_similarity(emb, kept_emb) >= t for kept_emb in kept_embeddings
        )
        if not is_duplicate:
            kept_chunks.append(chunk)
            kept_embeddings.append(emb)

    return kept_chunks
