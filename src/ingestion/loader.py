"""Step 1: Document loading from PDF, text, and markdown sources."""

from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document


def load_documents(source_path: str | Path) -> list[Document]:
    """Load documents from a file or directory."""
    path = Path(source_path)

    if path.is_file():
        return _load_single_file(path)

    if not path.is_dir():
        raise FileNotFoundError(f"Source path does not exist: {path}")

    documents: list[Document] = []

    pdf_loader = DirectoryLoader(
        str(path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    documents.extend(pdf_loader.load())

    txt_loader = DirectoryLoader(
        str(path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents.extend(txt_loader.load())

    md_loader = DirectoryLoader(
        str(path),
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
    )
    documents.extend(md_loader.load())

    if not documents:
        raise ValueError(f"No supported documents found in {path}")

    return documents


def _load_single_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix == ".txt":
        return TextLoader(str(path), encoding="utf-8").load()
    if suffix == ".md":
        return UnstructuredMarkdownLoader(str(path)).load()
    raise ValueError(f"Unsupported file type: {suffix}")
