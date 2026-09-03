"""
chunking.py
===========

Splits extracted page text (from ingestion.py) into smaller, overlapping
chunks suitable for embedding and retrieval.

WHY CHUNKING IS NEEDED
------------------------
Embedding models and retrieval work best on small, topically-focused
pieces of text:

1. A whole page (or whole paper) embedded as ONE vector would blur
   together many different ideas into a single point in vector space.
   A query about one specific idea on that page would retrieve the
   entire page, most of which is irrelevant — wasting LLM context and
   diluting the signal the model needs to answer accurately.

2. LLMs have a limited context window. Retrieving whole pages/documents
   for every query would blow through that budget almost immediately
   once more than a couple of documents are indexed.

WHY OVERLAP IS NEEDED
------------------------
If chunk boundaries are chosen arbitrarily, an important sentence or
idea can be split in half — e.g. "RAG-Token differs from RAG-Sequence
in that [PAGE/CHUNK BREAK] it marginalizes at the token level rather
than the sequence level." Neither resulting chunk alone answers a
question about this distinction. Overlap (chunk_overlap characters
repeated at the start of the next chunk) makes it far less likely that
a complete idea is split across two "disconnected" chunks.

WHAT WOULD HAPPEN WITHOUT PROPER CHUNKING
--------------------------------------------
- No chunking (whole-document embeddings): poor retrieval precision.
- Naive fixed-length chunking with no respect for text structure
  (breaking mid-word, mid-sentence at hard character boundaries):
  chunks that don't read as coherent units, confusing the LLM.

We use LangChain's RecursiveCharacterTextSplitter, which tries to
split on paragraph breaks first, then sentence breaks, then words,
only falling back to hard character cuts as a last resort — this
keeps chunks as semantically coherent as possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.ingestion import PageDocument


@dataclass
class Chunk:
    """
    A single retrievable unit: a slice of text plus full citation
    metadata inherited from the page it came from.
    """

    text: str
    source: str
    page: int
    document_id: str
    chunk_id: str   # globally unique across the whole knowledge base

    def metadata(self, content_hash: str | None = None) -> dict:
        """
        Metadata dict in the shape ChromaDB expects (flat, JSON-safe).

        content_hash (V2): a hash of the source PDF's bytes, stamped
        onto every chunk from that document. Lets vector_store.py
        detect "same filename, different content" and re-index rather
        than silently skip — see VectorStore.get_document_hashes().
        """
        meta = {
            "source": self.source,
            "page": self.page,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
        }
        if content_hash:
            meta["content_hash"] = content_hash
        return meta


def chunk_pages(
    pages: list[PageDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split a list of PageDocuments into Chunks.

    We chunk PER PAGE (not by concatenating the whole document first)
    so that every chunk unambiguously belongs to exactly one page
    number — this keeps citations accurate ("Page 4" really means
    page 4 of the PDF, not an approximation).

    Args:
        pages: output of ingestion.load_pdf(s).
        chunk_size: max characters per chunk.
        chunk_overlap: characters shared between consecutive chunks.

    Returns:
        A flat list of Chunk objects, each with a unique chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try paragraph breaks first, then lines, then sentences, then
        # words, only falling back to raw characters as a last resort.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    running_index = 0  # guarantees unique, deterministic chunk IDs

    for page in pages:
        pieces = splitter.split_text(page.text)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = f"{page.document_id}_p{page.page}_c{running_index}"
            chunks.append(
                Chunk(
                    text=piece,
                    source=page.source,
                    page=page.page,
                    document_id=page.document_id,
                    chunk_id=chunk_id,
                )
            )
            running_index += 1

    return chunks
