"""
Hybrid retriever factory.

Combines a BM25 keyword retriever (nails exact "Article 21" / "Schedule 7" lookups)
with a Chroma vector retriever (conceptual queries) via an EnsembleRetriever.

The vector store is built offline by ``app.ingest``; this module only *loads* it.
BM25 has no persisted index, so it is rebuilt at startup from ``chunks.jsonl`` (the
same chunks that were embedded) — no re-embedding required. Query-time embeddings are
computed by a local Ollama model (``qwen3-embedding:4b``); Ollama is assumed running.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from app.config import get_settings


def _load_chunks(path: str) -> list[Document]:
    docs: list[Document] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(Document(page_content=rec["page_content"], metadata=rec["metadata"]))
    return docs


def build_retriever():
    """Load the persisted vector store + rebuild BM25, return a hybrid EnsembleRetriever."""
    settings = get_settings()

    if not Path(settings.chroma_persist_dir).exists() or not Path(settings.chunks_path).exists():
        raise RuntimeError(
            f"Vector store not found at {settings.chroma_persist_dir!r}. "
            "Run `uv run python -m app.ingest` first."
        )

    # Same local Ollama model used at ingest — embedding is symmetric (no task_type),
    # so query and document vectors live in the same space.
    embeddings = OllamaEmbeddings(model=settings.embedding_model)
    vectorstore = Chroma(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})

    bm25 = BM25Retriever.from_documents(_load_chunks(settings.chunks_path))
    bm25.k = settings.retrieval_k

    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[settings.bm25_weight, settings.vector_weight],
    )
