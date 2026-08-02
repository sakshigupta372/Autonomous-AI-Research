"""ChromaDB-backed vector memory for paper chunks (episodic long-term memory)."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from research_agent.models.paper import Chunk

COLLECTION_NAME = "paper_chunks"


def get_collection(chroma_dir: Path):
    """Get (or create) the chunk collection, embedded locally via ChromaDB's
    bundled ONNX MiniLM model. Free and offline after the one-time model
    download; no API key required.
    """
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_fn = DefaultEmbeddingFunction()
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def add_chunks(collection, chunks: list[Chunk]) -> None:
    """Embed and upsert chunks into the vector store, keyed by a stable chunk id."""
    if not chunks:
        return
    ids = [f"{c.paper_id}:{c.section}:{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [{"paper_id": c.paper_id, "section": c.section} for c in chunks]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def query(collection, text: str, top_k: int = 5) -> list[str]:
    """Semantic search over stored chunks, returning matching document texts."""
    results = collection.query(query_texts=[text], n_results=top_k)
    return results.get("documents", [[]])[0]
