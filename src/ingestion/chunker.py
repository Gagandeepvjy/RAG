"""Step 2: Flat chunking with RecursiveCharacterTextSplitter."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks, returning LangChain Document objects."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
