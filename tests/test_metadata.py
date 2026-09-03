"""
Tests verifying metadata (source, page, chunk_id, document_id) survives
every stage: chunking -> vector store -> retrieval.

This directly tests the V2 requirement: "A retrieved document should
contain enough information to determine source, page, chunk_id,
document_id."
"""

from unittest.mock import MagicMock

from app.chunking import chunk_pages
from app.ingestion import PageDocument
from app.retriever import Retriever
from app.vector_store import VectorStore


def test_metadata_dict_contains_all_required_fields():
    page = PageDocument(text="Some content about RAG.", source="RAG.pdf", page=4, document_id="rag_pdf")
    chunks = chunk_pages([page])

    meta = chunks[0].metadata()
    assert meta["source"] == "RAG.pdf"
    assert meta["page"] == 4
    assert meta["document_id"] == "rag_pdf"
    assert "chunk_id" in meta and meta["chunk_id"]


def test_content_hash_included_when_provided():
    page = PageDocument(text="Some content.", source="RAG.pdf", page=1, document_id="rag_pdf")
    chunk = chunk_pages([page])[0]

    meta_without_hash = chunk.metadata()
    meta_with_hash = chunk.metadata(content_hash="abc123")

    assert "content_hash" not in meta_without_hash
    assert meta_with_hash["content_hash"] == "abc123"


def test_metadata_survives_full_round_trip_through_vector_store_and_retrieval():
    """
    PDF page -> Chunk -> VectorStore -> Retriever -> RetrievedChunk
    Every metadata field must still be correct at the end.
    """
    store = VectorStore(collection_name="test_metadata_collection")
    store.reset()

    page = PageDocument(
        text="RAG-Token marginalizes over documents at each generation step.",
        source="RAG Literature Review.pdf",
        page=12,
        document_id="rag_literature_review_pdf",
    )
    chunks = chunk_pages([page])
    chunk = chunks[0]

    store.add_chunks(
        ids=[chunk.chunk_id],
        texts=[chunk.text],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[chunk.metadata(content_hash="deadbeef")],
    )

    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
    retriever = Retriever(store, fake_embedder)

    results = retriever.retrieve("What is RAG-Token?", top_k=1)

    assert len(results) == 1
    retrieved = results[0]
    assert retrieved.source == "RAG Literature Review.pdf"
    assert retrieved.page == 12
    assert retrieved.document_id == "rag_literature_review_pdf"
    assert retrieved.chunk_id == chunk.chunk_id

    store.reset()


def test_metadata_preserved_across_multiple_documents():
    store = VectorStore(collection_name="test_metadata_multi_collection")
    store.reset()

    pages = [
        PageDocument(text="RAG content here.", source="RAG.pdf", page=1, document_id="rag_pdf"),
        PageDocument(text="BERT content here.", source="BERT.pdf", page=3, document_id="bert_pdf"),
    ]
    all_chunks = []
    for p in pages:
        all_chunks.extend(chunk_pages([p]))

    store.add_chunks(
        ids=[c.chunk_id for c in all_chunks],
        texts=[c.text for c in all_chunks],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[c.metadata() for c in all_chunks],
    )

    registry = store.get_document_registry()
    assert registry == {"rag_pdf": "RAG.pdf", "bert_pdf": "BERT.pdf"}

    store.reset()
