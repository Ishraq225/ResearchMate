"""
pipeline.py
===========

Orchestrates the "process documents" workflow:
    PDFs -> ingestion -> chunking -> embeddings -> vector store

WHY THIS EXISTS SEPARATELY FROM app.py
-------------------------------------------
app.py's job is to be a thin UI layer: draw widgets, react to clicks,
display results. If the ingest -> chunk -> embed -> store sequence
lived directly inside a Streamlit button callback, it would be
impossible to test without spinning up a browser, and impossible to
reuse (e.g. from evaluation/evaluate.py, which also needs to index
documents). Putting it here makes the pipeline a plain Python function
that both the UI and the evaluation script can call identically.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from app.chunking import chunk_pages
from app.embeddings import EmbeddingManager
from app.ingestion import IngestionError, load_pdf
from app.vector_store import VectorStore


@dataclass
class IndexingReport:
    """Summary returned after processing a batch of PDFs, shown in the UI."""

    files_processed: list[str] = field(default_factory=list)
    files_reprocessed: list[str] = field(default_factory=list)  # content changed, re-indexed
    files_skipped: list[str] = field(default_factory=list)        # unchanged, already indexed
    files_failed: dict = field(default_factory=dict)               # filename -> error message
    chunks_added: int = 0


def _hash_file(path: Path) -> str:
    """SHA-256 hash of a file's raw bytes — used to detect content changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_documents(
    file_paths: list[str | Path],
    vector_store: VectorStore,
    embedding_manager: EmbeddingManager,
) -> IndexingReport:
    """
    Index a batch of PDF files into the vector store.

    V2 performance/correctness improvement over V1: rather than
    skipping re-indexing purely because a document_id (derived from
    filename) is already present, we compare a SHA-256 hash of the
    file's bytes against the hash stored on its existing chunks:

    - same document_id, same hash  -> skip (truly unchanged)
    - same document_id, diff hash  -> delete old chunks, re-index
    - new document_id              -> index normally

    This satisfies "avoid re-indexing unchanged documents" while also
    catching the case V1 missed: a file re-uploaded under the same
    name with different content.
    """
    report = IndexingReport()
    existing_hashes = vector_store.get_document_hashes()

    for path in file_paths:
        path = Path(path)
        filename = path.name

        try:
            content_hash = _hash_file(path)
        except OSError as exc:
            report.files_failed[filename] = f"Couldn't read file: {exc}"
            continue

        try:
            pages = load_pdf(path, original_filename=filename)
        except IngestionError as exc:
            report.files_failed[filename] = str(exc)
            continue

        document_id = pages[0].document_id
        previously_indexed_hash = existing_hashes.get(document_id)

        if previously_indexed_hash == content_hash:
            report.files_skipped.append(filename)
            continue

        chunks = chunk_pages(pages)
        if not chunks:
            report.files_failed[filename] = "No chunks produced (document may be empty)."
            continue

        if previously_indexed_hash is not None:
            # Content changed under the same document_id — clear stale
            # chunks first so old and new text don't coexist.
            vector_store.delete_document(document_id)

        texts = [c.text for c in chunks]
        embeddings = embedding_manager.embed_documents(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata(content_hash=content_hash) for c in chunks]

        vector_store.add_chunks(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        if previously_indexed_hash is not None:
            report.files_reprocessed.append(filename)
        else:
            report.files_processed.append(filename)
        report.chunks_added += len(chunks)
        existing_hashes[document_id] = content_hash

    return report
