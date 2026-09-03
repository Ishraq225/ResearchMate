"""
Tests for app/pipeline.py: process_documents' content-hash-based
skip / re-index / duplicate handling.

Uses real PDFs (generated on the fly with reportlab, a lightweight
dependency already available in most Python environments) and a REAL
VectorStore, but a MOCKED EmbeddingManager so tests run offline and
fast without downloading the sentence-transformers model.
"""

from unittest.mock import MagicMock

import pytest

from app.pipeline import process_documents
from app.vector_store import VectorStore


def _make_pdf(path, lines: list[str]):
    reportlab = pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 15
    c.showPage()
    c.save()


@pytest.fixture
def fake_embedder():
    embedder = MagicMock()
    embedder.embed_documents.side_effect = lambda texts, **kw: [[0.1, 0.2, 0.3] for _ in texts]
    return embedder


@pytest.fixture
def store():
    vs = VectorStore(collection_name="test_pipeline_collection")
    vs.reset()
    yield vs
    vs.reset()


def test_new_document_is_processed(tmp_path, store, fake_embedder):
    pdf_path = tmp_path / "RAG.pdf"
    _make_pdf(pdf_path, ["Retrieval-Augmented Generation combines retrieval and generation."])

    report = process_documents([pdf_path], vector_store=store, embedding_manager=fake_embedder)

    assert "RAG.pdf" in report.files_processed
    assert report.chunks_added > 0
    assert store.count() > 0


def test_unchanged_document_is_skipped_on_second_run(tmp_path, store, fake_embedder):
    pdf_path = tmp_path / "RAG.pdf"
    _make_pdf(pdf_path, ["Same content every time."])

    process_documents([pdf_path], vector_store=store, embedding_manager=fake_embedder)
    chunks_after_first_run = store.count()

    report_second = process_documents([pdf_path], vector_store=store, embedding_manager=fake_embedder)

    assert "RAG.pdf" in report_second.files_skipped
    assert report_second.files_processed == []
    assert store.count() == chunks_after_first_run  # nothing duplicated


def test_changed_content_under_same_filename_is_reprocessed(tmp_path, store, fake_embedder):
    pdf_path = tmp_path / "RAG.pdf"

    _make_pdf(pdf_path, ["Original content version one."])
    process_documents([pdf_path], vector_store=store, embedding_manager=fake_embedder)

    # Overwrite the SAME filename with DIFFERENT content.
    _make_pdf(pdf_path, ["Completely different content version two, much longer than before."])
    report = process_documents([pdf_path], vector_store=store, embedding_manager=fake_embedder)

    assert "RAG.pdf" in report.files_reprocessed
    assert "RAG.pdf" not in report.files_skipped

    # Old content should no longer be present in the store.
    hits = store.similarity_search([0.1, 0.2, 0.3], top_k=10)
    texts = [h["text"] for h in hits]
    assert not any("version one" in t for t in texts)


def test_invalid_pdf_is_reported_as_failure(tmp_path, store, fake_embedder):
    bad_path = tmp_path / "not_a_pdf.pdf"
    bad_path.write_text("this is definitely not PDF content")

    report = process_documents([bad_path], vector_store=store, embedding_manager=fake_embedder)

    assert "not_a_pdf.pdf" in report.files_failed
    assert report.files_processed == []


def test_multiple_pdfs_are_all_processed(tmp_path, store, fake_embedder):
    pdf1 = tmp_path / "RAG.pdf"
    pdf2 = tmp_path / "BERT.pdf"
    _make_pdf(pdf1, ["RAG paper content."])
    _make_pdf(pdf2, ["BERT paper content."])

    report = process_documents([pdf1, pdf2], vector_store=store, embedding_manager=fake_embedder)

    assert set(report.files_processed) == {"RAG.pdf", "BERT.pdf"}
    assert len(store.get_document_registry()) == 2
