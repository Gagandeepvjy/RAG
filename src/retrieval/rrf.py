"""Reciprocal Rank Fusion utility."""

from src.models import DocumentChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists using RRF.

    Each list contains chunk_ids ordered by relevance (best first).
    Returns merged list of (chunk_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def weighted_reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge ranked lists using weighted RRF.

    Each ranked list receives a weight, so the final score is a weighted sum
    of reciprocal rank contributions.
    """
    if len(ranked_lists) != len(weights):
        raise ValueError("ranked_lists and weights must have the same length")

    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def to_chunk_list(
    fused: list[tuple[str, float]],
    chunk_lookup: dict[str, DocumentChunk],
) -> list[DocumentChunk]:
    """Convert fused chunk_id scores to DocumentChunk objects."""
    return [chunk_lookup[cid] for cid, _ in fused if cid in chunk_lookup]
