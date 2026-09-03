"""
retriever.py
============

The retrieval layer: given a natural-language query, returns the
top-K most relevant chunks from the vector store, with scores and
metadata attached.

WHY THIS IS ITS OWN MODULE (SEPARATE FROM vector_store.py)
-----------------------------------------------------------------
vector_store.py knows about ChromaDB internals (collections, upsert,
query calls). retriever.py knows about the RAG *concept* of
retrieval: turn a query into an embedding, ask the store for matches,
shape the results into something generator.py and app.py can use
without needing to know anything about ChromaDB or embeddings.

This separation means:
- We could swap ChromaDB for another vector DB and only vector_store.py
  changes.
- The UI (app.py) never talks to ChromaDB or the embedding model
  directly — it only ever calls retriever.retrieve(query). This keeps
  the retrieval layer independent from the UI, as required.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import TOP_K
from app.embeddings import EmbeddingManager
from app.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    """One retrieved chunk, ready to be shown to the user or fed to the LLM."""

    text: str
    source: str
    page: int
    document_id: str
    chunk_id: str
    distance: float  # cosine distance: 0 = identical, 2 = opposite

    @property
    def similarity(self) -> float:
        """
        Convert distance to a more intuitive 0-1 similarity score for
        display purposes (1 = perfect match, 0 = unrelated).
        Cosine distance ranges [0, 2], so similarity = 1 - distance/2.
        """
        return max(0.0, min(1.0, 1 - self.distance / 2))


class Retriever:
    """
    Ties together EmbeddingManager (query -> vector) and VectorStore
    (vector -> nearest chunks) into a single `retrieve()` call.
    """

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Args:
            document_ids: OPTIONAL allow-list restricting retrieval to
                specific documents (V2 "document selection" feature).
                None means "search everything indexed". An empty list
                means "the user deselected every document" and
                deliberately returns nothing — see vector_store.py's
                similarity_search docstring for why this distinction
                is made at the database layer, not after the fact.

        Steps (mirrors the RAG retrieval diagram in the project spec):
            query -> query embedding -> ChromaDB search -> top-K chunks
        """
        query = (query or "").strip()
        if not query:
            return []

        query_embedding = self.embedding_manager.embed_query(query)
        raw_hits = self.vector_store.similarity_search(
            query_embedding, top_k=top_k, document_ids=document_ids
        )

        chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()
        for hit in raw_hits:
            meta = hit["metadata"]
            chunk_id = meta.get("chunk_id", "unknown")
            # Defensive de-duplication: ChromaDB's upsert-by-id already
            # prevents duplicate rows, but if a document was ever
            # processed under two different document_ids (e.g. renamed
            # and re-uploaded), the same passage could appear twice in
            # results. We keep only the first (best-ranked) occurrence.
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            chunks.append(
                RetrievedChunk(
                    text=hit["text"],
                    source=meta.get("source", "unknown"),
                    page=meta.get("page", -1),
                    document_id=meta.get("document_id", "unknown"),
                    chunk_id=chunk_id,
                    distance=hit["distance"],
                )
            )
        return chunks
