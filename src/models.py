"""Shared data models."""

from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """A child chunk used for indexing and retrieval."""

    chunk_id: str
    text: str
    source: str
    parent_id: str = ""          # ID of the parent chunk this belongs to
    metadata: dict = field(default_factory=dict)


@dataclass
class ParentChunk:
    """A large parent chunk returned as context to the LLM."""

    parent_id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)
