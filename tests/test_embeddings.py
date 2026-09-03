"""
Tests for app/embeddings.py.

We avoid hitting the network / downloading the real sentence-transformers
model in tests by injecting a fake model object that mimics the
SentenceTransformer.encode() interface. This keeps tests fast,
deterministic, and runnable offline/in CI.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.embeddings import EmbeddingError, EmbeddingManager


def _manager_with_fake_model(encode_side_effect=None) -> EmbeddingManager:
    """Build an EmbeddingManager whose underlying model is a mock."""
    manager = EmbeddingManager.__new__(EmbeddingManager)  # skip __init__ (avoids real model load)
    manager.model_name = "fake-model"
    manager.model = MagicMock()
    if encode_side_effect is not None:
        manager.model.encode.side_effect = encode_side_effect
    return manager


def test_embed_documents_returns_list_of_vectors():
    fake_vectors = np.array([[0.1, 0.2], [0.3, 0.4]])
    manager = _manager_with_fake_model(lambda *a, **kw: fake_vectors)

    result = manager.embed_documents(["hello", "world"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_documents_empty_list_returns_empty():
    manager = _manager_with_fake_model()
    assert manager.embed_documents([]) == []


def test_embed_query_returns_single_vector():
    fake_vector = np.array([0.5, 0.6, 0.7])
    manager = _manager_with_fake_model(lambda *a, **kw: fake_vector)

    result = manager.embed_query("what is RAG?")

    assert result == [0.5, 0.6, 0.7]


def test_embed_query_rejects_empty_string():
    manager = _manager_with_fake_model()
    with pytest.raises(EmbeddingError):
        manager.embed_query("")


def test_embed_query_rejects_whitespace_only():
    manager = _manager_with_fake_model()
    with pytest.raises(EmbeddingError):
        manager.embed_query("   ")


def test_embed_documents_wraps_model_failures():
    def _boom(*a, **kw):
        raise RuntimeError("model exploded")

    manager = _manager_with_fake_model(_boom)
    with pytest.raises(EmbeddingError):
        manager.embed_documents(["some text"])
