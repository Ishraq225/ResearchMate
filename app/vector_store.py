"""
vector_store.py
================

Manages the ChromaDB collection: creating/loading it, adding chunk
vectors + metadata, running similarity search, and resetting it.

WHY A DEDICATED VECTOR DATABASE (INSTEAD OF, SAY, A PYTHON LIST)
---------------------------------------------------------------------
We could store embeddings in memory (a Python list of vectors) and do
similarity search with brute-force numpy. That works for a demo but:
1. It doesn't persist — restart the app, lose everything, re-embed
   every document from scratch (slow, and wasteful of compute).
2. It doesn't scale — brute force cosine similarity over thousands of
   chunks in pure Python gets slow; a real vector DB uses indexing
   structures (e.g. HNSW) designed for fast approximate search.
3. It doesn't give us metadata filtering, persistence, or a stable
   API — all of which ChromaDB provides out of the box.

WHY PERSISTENCE MATTERS HERE SPECIFICALLY
----------------------------------------------
Users expect to upload papers once and keep asking questions across
multiple app sessions without waiting for re-indexing every time.
ChromaDB's PersistentClient writes the index to disk
(data/vector_store/), so a restart loads the existing collection
instead of starting empty.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_COLLECTION_NAME, VECTOR_STORE_DIR


class VectorStoreError(Exception):
    """Raised when the vector database fails to read/write data."""


class VectorStore:
    """
    Thin wrapper around a persistent ChromaDB collection.

    Responsibilities (per project spec):
    - create/load a collection
    - add documents (chunk text + embeddings + metadata)
    - persist vectors locally
    - similarity search
    - delete/reset the collection
    - report basic stats (chunk count, indexed document IDs)
    """

    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME):
        self.collection_name = collection_name
        try:
            self._client = chromadb.PersistentClient(
                path=str(VECTOR_STORE_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            # get_or_create_collection is what gives us restart-safe
            # persistence: if the collection already exists on disk,
            # we load it instead of creating an empty one.
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to initialize ChromaDB: {exc}") from exc

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def add_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """
        Add (or upsert) chunk vectors into the collection.

        We use `upsert` rather than `add` so that re-processing the
        same document (same chunk_ids) overwrites rather than
        duplicates entries — important for the "avoid re-indexing
        unchanged documents" performance requirement, and for making
        re-runs idempotent.
        """
        if not ids:
            return
        try:
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to add {len(ids)} chunk(s) to the vector store: {exc}") from exc

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def similarity_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """
        Retrieve the top_k most semantically similar chunks to a query
        embedding.

        Returns a list of dicts, each shaped like:
            {
                "text": str,
                "metadata": {"source": ..., "page": ..., "chunk_id": ...},
                "distance": float,   # lower = more similar (cosine distance)
            }
        """
        if self.count() == 0:
            return []
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.count()),
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        hits: list[dict] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            hits.append({"text": text, "metadata": metadata, "distance": distance})
        return hits

    def count(self) -> int:
        """Number of chunks currently stored."""
        try:
            return self._collection.count()
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to count chunks: {exc}") from exc

    def get_indexed_document_ids(self) -> set[str]:
        """
        Returns the set of document_ids already present in the store.

        Used to skip re-embedding a document that's already indexed
        (see app.py's "avoid recomputing embeddings" logic).
        """
        if self.count() == 0:
            return set()
        try:
            data = self._collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to read indexed documents: {exc}") from exc

        return {m["document_id"] for m in data.get("metadatas", []) if m and "document_id" in m}

    # ------------------------------------------------------------------
    # Destructive operations
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Delete the entire collection and recreate it empty.

        This is the "Clear Knowledge Base" action in the UI — it must
        be explicit and user-triggered, never automatic, since it's
        destructive and not reversible.
        """
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to reset the vector store: {exc}") from exc
