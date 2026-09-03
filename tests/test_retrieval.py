"""
Tests for app/retriever.py, including V2's document-selection filtering.

Uses a REAL VectorStore (ChromaDB, ephemeral test collection) with a
mocked EmbeddingManager, so we test actual retrieval/filtering
behavior rather than mocking the database away entirely.
"""

from unittest.mock import MagicMock

import pytest

from app.retriever import Retriever
from app.vector_store import VectorStore


@pytest.fixture
def populated_store():
    """A VectorStore with 3 chunks across 2 documents, reset before and after."""
    store = VectorStore(collection_name="test_retrieval_collection")
    store.reset()

    store.add_chunks(
        ids=["a", "b", "c"],
        texts=[
            "RAG-Token marginalizes over documents at each generated token.",
            "BERT is pretrained with masked language modeling.",
            "RAG-Sequence uses one retrieved document for the whole output.",
        ],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
        metadatas=[
            {"source": "RAG.pdf", "page": 2, "document_id": "rag_pdf", "chunk_id": "a"},
            {"source": "BERT.pdf", "page": 1, "document_id": "bert_pdf", "chunk_id": "b"},
            {"source": "RAG.pdf", "page": 5, "document_id": "rag_pdf", "chunk_id": "c"},
        ],
    )
    yield store
    store.reset()


def _fake_embedder(query_vector):
    embedder = MagicMock()
    embedder.embed_query.return_value = query_vector
    return embedder


def test_retrieve_returns_relevant_chunks_ranked(populated_store):
    embedder = _fake_embedder([1.0, 0.0, 0.0])  # closest to chunk 'a'
    retriever = Retriever(populated_store, embedder)

    results = retriever.retrieve("What is RAG-Token?", top_k=3)

    assert len(results) == 3
    assert results[0].chunk_id == "a"  # best match ranked first


def test_document_filtering_restricts_results_to_selected_documents(populated_store):
    embedder = _fake_embedder([0.0, 1.0, 0.0])  # would normally match BERT.pdf best
    retriever = Retriever(populated_store, embedder)

    # Restrict to only RAG.pdf -> BERT chunk must never appear, even
    # though it's the closest vector overall.
    results = retriever.retrieve("test query", top_k=3, document_ids=["rag_pdf"])

    assert len(results) == 2
    assert all(r.document_id == "rag_pdf" for r in results)


def test_empty_document_selection_returns_no_results(populated_store):
    embedder = _fake_embedder([1.0, 0.0, 0.0])
    retriever = Retriever(populated_store, embedder)

    # Explicitly deselecting every document (empty list, not None) must
    # return nothing, distinct from "no filter applied".
    results = retriever.retrieve("test query", top_k=3, document_ids=[])
    assert results == []


def test_retrieve_on_empty_store_returns_empty_list():
    store = VectorStore(collection_name="test_retrieval_empty_collection")
    store.reset()
    embedder = _fake_embedder([1.0, 0.0, 0.0])
    retriever = Retriever(store, embedder)

    results = retriever.retrieve("anything", top_k=5)
    assert results == []


def test_retrieve_with_blank_query_returns_empty_list(populated_store):
    embedder = _fake_embedder([1.0, 0.0, 0.0])
    retriever = Retriever(populated_store, embedder)

    assert retriever.retrieve("   ", top_k=3) == []
    # embed_query should never have been called for a blank query
    embedder.embed_query.assert_not_called()
