"""Application configuration."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # HuggingFace
    huggingfacehub_api_token: str = ""
    hf_embedding_model: str = "BAAI/bge-large-en-v1.5"

    # Cohere
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"

    # Ollama (LLM + query expansion)
    ollama_url: str = ""
    ollama_chat_model: str = "qwen3:8b"
    ollama_expansion_model: str = "qwen3:8b"

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # Retrieval
    num_query_expansions: int = 3
    retrieval_top_k: int = 20
    rerank_top_k: int = 10
    rrf_k: int = 60
    vector_weight: float = 0.7
    bm25_weight: float = 0.3

    # Deduplication — cosine similarity threshold above which two chunks are
    # considered near-duplicates (the lower-ranked one is dropped)
    dedup_threshold: float = 0.95

    # Paths
    index_dir: Path = Path(".index")
    documents_dir: Path = Path("data/documents")
    hf_home: Path = Path(".cache/huggingface")

    # Company-specific boosting
    company_boost: bool = True
    company_boost_min_token_len: int = 3

    # Conversation history length (number of past QA turns to keep)
    conversation_history_len: int = 5

    def model_post_init(self, __context) -> None:
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        if self.huggingfacehub_api_token:
            os.environ.setdefault(
                "HUGGINGFACEHUB_API_TOKEN", self.huggingfacehub_api_token
            )

    @property
    def chroma_path(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def parent_chunks_path(self) -> Path:
        return self.index_dir / "parent_chunks.json"

    @property
    def ingestion_manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"


settings = Settings()
