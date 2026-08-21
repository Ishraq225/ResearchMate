"""
embeddings.py
=============

Converts text (document chunks AND user queries) into numerical
vectors ("embeddings") that capture semantic meaning.

WHY EMBEDDINGS ARE NEEDED
----------------------------
Keyword search (e.g. simple string matching) fails when the user's
wording doesn't match the document's wording. E.g. a user asking
"How does RAG combine retrieval and generation?" should still match a
passage that says "the model conditions generation on retrieved
documents" even though almost no words overlap. Embeddings map text
into a high-dimensional space where SEMANTICALLY similar text ends up
CLOSE TOGETHER, regardless of exact wording. That's what makes
semantic search possible.

WHY DOCUMENTS AND QUERIES MUST SHARE ONE EMBEDDING SPACE
-------------------------------------------------------------
If chunks were embedded with one model and queries with a different
model (or even a different version/config of the "same" model), their
vectors would not be comparable — distances between them would be
meaningless. This module guarantees a SINGLE shared model instance is
used for both, via `EmbeddingManager`.

WHY WE CACHE THE MODEL
--------------------------
Loading a sentence-transformers model from disk takes real time
(parsing weights, moving to device). If we reloaded it on every
Streamlit rerun (which happens on every UI interaction), the app
would feel sluggish. We load it once per process and reuse it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME


class EmbeddingError(Exception):
    """Raised when the embedding model fails to load or encode text."""


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    """
    Load (and cache) the sentence-transformers model.

    lru_cache ensures that no matter how many times this function is
    called (e.g. across Streamlit reruns within the same process),
    the actual model is only loaded into memory ONCE per model_name.
    """
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(
            f"Failed to load embedding model '{model_name}'. "
            "Check your internet connection (first run downloads the "
            "model) or the model name in your .env file."
        ) from exc


class EmbeddingManager:
    """
    Thin wrapper around a SentenceTransformer model.

    Responsibilities (per project spec):
    1. Load the embedding model.
    2. Convert document chunks into vectors.
    3. Convert user queries into vectors.
    4. Guarantee both use the exact same embedding space.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = _load_model(model_name)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed a batch of document chunk texts.

        Batching (rather than one-encode-call-per-chunk) is a real
        performance optimization: the model can process many texts
        in parallel on CPU/GPU far more efficiently than one at a time.
        """
        if not texts:
            return []
        try:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to embed {len(texts)} chunk(s): {exc}") from exc
        return _to_list(vectors)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single user query.

        Kept as a separate method (rather than reusing embed_documents
        with a list of one) purely for readability/intent at call
        sites — under the hood it uses the exact same model instance,
        which is what actually matters for the vectors to be comparable.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        try:
            vector = self.model.encode(text, convert_to_numpy=True)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc
        return vector.tolist()


def _to_list(vectors: np.ndarray) -> list[list[float]]:
    """Convert a numpy array of vectors into plain Python lists (JSON/Chroma-safe)."""
    return [v.tolist() for v in vectors]
