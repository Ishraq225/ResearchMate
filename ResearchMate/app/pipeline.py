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
    files_skipped: list[str] = field(default_factory=list)  # already indexed
    files_failed: dict = field(default_factory=dict)         # filename -> error message
    chunks_added: int = 0


def process_documents(
    file_paths: list[str | Path],
    vector_store: VectorStore,
    embedding_manager: EmbeddingManager,
) -> IndexingReport:
    """
    Index a batch of PDF files into the vector store.

    Performance note: we skip re-embedding any document whose
    document_id is already present in the vector store — this
    satisfies the "avoid re-indexing unchanged documents" requirement.
    A document is considered "changed" only if its filename-derived ID
    isn't already indexed; V1 does not do content hashing/diffing.
    """
    report = IndexingReport()
    already_indexed = vector_store.get_indexed_document_ids()

    for path in file_paths:
        path = Path(path)
        filename = path.name

        try:
            pages = load_pdf(path, original_filename=filename)
        except IngestionError as exc:
            report.files_failed[filename] = str(exc)
            continue

        document_id = pages[0].document_id
        if document_id in already_indexed:
            report.files_skipped.append(filename)
            continue

        chunks = chunk_pages(pages)
        if not chunks:
            report.files_failed[filename] = "No chunks produced (document may be empty)."
            continue

        texts = [c.text for c in chunks]
        embeddings = embedding_manager.embed_documents(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata() for c in chunks]

        vector_store.add_chunks(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

        report.files_processed.append(filename)
        report.chunks_added += len(chunks)
        already_indexed.add(document_id)

    return report
