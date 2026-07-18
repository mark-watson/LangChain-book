"""Shared helpers for loading the tiny test corpus in ../data/."""

from pathlib import Path

from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent.parent / "data"


def load_documents() -> list[Document]:
    """Return one Document per .txt file in ../data/."""
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


TEST_QUESTIONS = [
    "Who tried to define what chemistry is?",
    "What is the Austrian School of Economics?",
    "How does body chemistry affect exercise?",
]
