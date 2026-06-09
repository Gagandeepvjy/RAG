"""Step 3: HuggingFace embeddings via sentence-transformers.

Model: BAAI/bge-large-en-v1.5 (state-of-the-art open-source retrieval model).
BGE models expect a query instruction prefix at query time for best results.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

# BGE models perform better with this instruction prefix on queries
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_embeddings: HuggingFaceEmbeddings | None = None


def get_embedder() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance."""
    global _embeddings
    # if _embeddings is None:
    #     _embeddings = HuggingFaceEmbeddings(
    #         model_name=settings.hf_embedding_model,
    #         model_kwargs={"device": "cpu"},
    #         encode_kwargs={"normalize_embeddings": True},
    #         # Prepend BGE query instruction at query time only
    #         if "bge" in settings.hf_embedding_model.lower()
    #         else "",
    #     )
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return _embeddings
