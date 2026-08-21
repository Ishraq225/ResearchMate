"""
Tests for app/ingestion.py.

Focused on error handling paths, since a robust ingestion layer is
mostly defined by how gracefully it fails on bad input.
"""

import pytest

from app.ingestion import IngestionError, _make_document_id, load_pdf


def test_missing_file_raises_ingestion_error(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(IngestionError):
        load_pdf(missing)


def test_non_pdf_file_raises_ingestion_error(tmp_path):
    fake_pdf = tmp_path / "not_really_a_pdf.pdf"
    fake_pdf.write_text("this is just plain text, not a PDF")
    with pytest.raises(IngestionError):
        load_pdf(fake_pdf)


def test_make_document_id_is_filesystem_safe():
    doc_id = _make_document_id("Retrieval Augmented Generation (2020).pdf")
    assert doc_id == "retrieval_augmented_generation_2020"


def test_make_document_id_never_empty():
    doc_id = _make_document_id("!!!.pdf")
    assert doc_id  # falls back to a generated uuid-based id
