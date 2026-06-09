"""Step 4: Dual-index storage (flat chunks) with incremental ingestion.

- Chunks are indexed in both Chroma (vector) and BM25 (keyword).
- No parent chunks — every chunk is indexed and served to the LLM directly.
- A manifest (SHA-256 of each file) tracks which files are already ingested
  so only new/changed documents are processed on subsequent runs.
"""

import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.ingestion.embedder import get_embedder
from src.models import DocumentChunk, ParentChunk

try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain.retrievers import BM25Retriever  # type: ignore[no-redef]


class DualIndexStore:
    """Manages Chroma vector store, BM25 index, and parent chunk store."""

    COLLECTION_NAME = "rag_chunks"

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or settings.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._embedder = get_embedder()
        self._vectorstore: Chroma | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._child_chunks: list[DocumentChunk] = []
        self._manifest: dict[str, str] = {}                # filepath → sha256

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def chunks(self) -> list[DocumentChunk]:
        """Child chunks (used by retrieval pipeline)."""
        return self._child_chunks

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            raise RuntimeError("Index not built/loaded yet.")
        return self._vectorstore

    @property
    def bm25_retriever(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            raise RuntimeError("BM25 index not built/loaded yet.")
        return self._bm25_retriever

    # ------------------------------------------------------------------
    # Incremental manifest helpers
    # ------------------------------------------------------------------

    def load_manifest(self) -> dict[str, str]:
        if settings.ingestion_manifest_path.exists():
            with open(settings.ingestion_manifest_path) as f:
                self._manifest = json.load(f)
        return self._manifest

    def save_manifest(self) -> None:
        with open(settings.ingestion_manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def file_needs_ingestion(self, filepath: Path) -> bool:
        """Return True if the file is new or has changed since last ingestion."""
        sha = _sha256(filepath)
        if self._manifest.get(str(filepath)) == sha:
            return False
        self._manifest[str(filepath)] = sha
        return True

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        parent_chunks: list[ParentChunk],  # ignored in flat mode
        child_chunks: list[DocumentChunk],
    ) -> None:
        """Build both indexes from scratch."""
        self._child_chunks = child_chunks

        lc_docs = self._to_lc_docs(child_chunks)

        # Chroma vector store
        self._vectorstore = Chroma.from_documents(
            documents=lc_docs,
            embedding=self._embedder,
            collection_name=self.COLLECTION_NAME,
            persist_directory=str(settings.chroma_path),
        )

        # BM25
        self._bm25_retriever = BM25Retriever.from_documents(lc_docs)
        self._bm25_retriever.k = settings.retrieval_top_k

        self._persist(child_chunks)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load indexes from disk."""
        if not settings.bm25_path.exists() or not settings.chunks_path.exists():
            raise FileNotFoundError(
                "Index not found. Run ingestion first: python scripts/ingest.py"
            )

        self._vectorstore = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=self._embedder,
            persist_directory=str(settings.chroma_path),
        )

        with open(settings.bm25_path, "rb") as f:
            self._bm25_retriever = pickle.load(f)

        with open(settings.chunks_path, encoding="utf-8") as f:
            self._child_chunks = [DocumentChunk(**item) for item in json.load(f)]

        self.load_manifest()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist(self, child_chunks: list[DocumentChunk]) -> None:
        with open(settings.bm25_path, "wb") as f:
            pickle.dump(self._bm25_retriever, f)
        with open(settings.chunks_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in child_chunks], f, indent=2)
        self.save_manifest()

    @staticmethod
    def _to_lc_docs(chunks: list[DocumentChunk]) -> list[Document]:
        return [
            Document(
                page_content=c.text,
                metadata={
                    "chunk_id": c.chunk_id,
                    "parent_id": c.parent_id,
                    "source": c.source,
                    **c.metadata,
                },
            )
            for c in chunks
        ]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
