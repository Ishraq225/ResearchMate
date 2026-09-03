"""
Tests for app/generator.py, focused on:
1. Context is passed correctly into the LLM prompt.
2. Insufficient evidence is detected and short-circuits WITHOUT
   calling the LLM provider at all.
"""

from unittest.mock import MagicMock

from app.generator import Generator, INSUFFICIENT_EVIDENCE_MESSAGE
from app.research_modes import ResearchMode
from app.retriever import RetrievedChunk


def _chunk(text="RAG combines retrieval and generation.", distance=0.1, source="RAG.pdf", page=1):
    return RetrievedChunk(
        text=text, source=source, page=page, document_id="rag_pdf", chunk_id="c1", distance=distance
    )


def test_context_is_passed_into_the_prompt():
    fake_provider = MagicMock()
    fake_provider.complete.return_value = "RAG combines retrieval with generation."
    generator = Generator(provider=fake_provider)

    chunk = _chunk(text="RAG-Token marginalizes at the token level.", distance=0.1)
    result = generator.generate_answer("What is RAG-Token?", [chunk])

    # The provider must have been called, and the chunk's text must
    # appear somewhere in the user prompt sent to it.
    fake_provider.complete.assert_called_once()
    _, user_prompt = fake_provider.complete.call_args[0]
    assert "RAG-Token marginalizes at the token level." in user_prompt
    assert "RAG.pdf" in user_prompt
    assert result.answer == "RAG combines retrieval with generation."
    assert result.insufficient_evidence is False


def test_insufficient_evidence_short_circuits_without_calling_llm():
    fake_provider = MagicMock()
    generator = Generator(provider=fake_provider)

    # No retrieved chunks at all -> definitely insufficient evidence.
    result = generator.generate_answer("What is RAG?", [])

    fake_provider.complete.assert_not_called()
    assert result.insufficient_evidence is True
    assert result.answer == INSUFFICIENT_EVIDENCE_MESSAGE


def test_low_similarity_chunks_trigger_insufficient_evidence():
    fake_provider = MagicMock()
    generator = Generator(provider=fake_provider)

    # cosine distance close to 2.0 -> similarity close to 0 -> below threshold
    weak_chunk = _chunk(distance=1.95)
    result = generator.generate_answer("Unrelated question", [weak_chunk])

    fake_provider.complete.assert_not_called()
    assert result.insufficient_evidence is True


def test_strong_match_does_not_trigger_insufficient_evidence():
    fake_provider = MagicMock()
    fake_provider.complete.return_value = "An answer."
    generator = Generator(provider=fake_provider)

    strong_chunk = _chunk(distance=0.05)  # similarity near 1.0
    result = generator.generate_answer("A relevant question", [strong_chunk])

    fake_provider.complete.assert_called_once()
    assert result.insufficient_evidence is False


def test_research_mode_changes_the_system_prompt():
    fake_provider = MagicMock()
    fake_provider.complete.return_value = "Summary text."
    generator = Generator(provider=fake_provider)

    chunk = _chunk(distance=0.05)
    result = generator.generate_answer("", [chunk], mode=ResearchMode.SUMMARIZE)

    assert "SUMMARIZE" in result.system_prompt_used
    assert "Key Contributions" in result.system_prompt_used
