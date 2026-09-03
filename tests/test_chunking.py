"""
Tests for app/chunking.py.

We test chunking in isolation (no PDFs, no embeddings, no network)
by constructing PageDocument objects directly. This keeps the test
fast and deterministic — exactly what unit tests should be.
"""

from app.chunking import chunk_pages
from app.ingestion import PageDocument


def _make_page(text: str, page: int = 1, source: str = "test.pdf", document_id: str = "test_pdf") -> PageDocument:
    return PageDocument(text=text, source=source, page=page, document_id=document_id)


def test_chunk_pages_produces_chunks_with_metadata():
    page = _make_page("Sentence one. " * 100)  # long enough to force multiple chunks
    chunks = chunk_pages([page], chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.source == "test.pdf"
        assert chunk.page == 1
        assert chunk.document_id == "test_pdf"
        assert chunk.chunk_id  # non-empty
        assert chunk.text.strip()


def test_chunk_ids_are_unique_across_pages():
    page1 = _make_page("Alpha content here. " * 50, page=1)
    page2 = _make_page("Beta content here. " * 50, page=2)
    chunks = chunk_pages([page1, page2], chunk_size=200, chunk_overlap=50)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # no duplicates


def test_short_text_produces_single_chunk():
    page = _make_page("Just a short sentence.")
    chunks = chunk_pages([page], chunk_size=800, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == "Just a short sentence."


def test_empty_pages_produce_no_chunks():
    chunks = chunk_pages([], chunk_size=800, chunk_overlap=150)
    assert chunks == []


def test_metadata_dict_shape():
    page = _make_page("Some text.")
    chunks = chunk_pages([page])
    meta = chunks[0].metadata()
    assert set(meta.keys()) == {"source", "page", "document_id", "chunk_id"}
