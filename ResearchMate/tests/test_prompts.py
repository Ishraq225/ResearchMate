"""Tests for app/prompts.py."""

from app.prompts import build_rag_prompt


def test_prompt_includes_context_and_question():
    chunks = [{"text": "RAG combines retrieval and generation.", "source": "RAG.pdf", "page": 2, "chunk_id": "x"}]
    prompt = build_rag_prompt("What is RAG?", chunks)

    assert "What is RAG?" in prompt
    assert "RAG.pdf" in prompt
    assert "RAG combines retrieval and generation." in prompt


def test_prompt_handles_empty_context_gracefully():
    prompt = build_rag_prompt("What is RAG?", [])
    assert "No relevant context was found" in prompt


def test_prompt_includes_chat_history_when_present():
    history = [("What is RAG?", "RAG stands for Retrieval-Augmented Generation.")]
    prompt = build_rag_prompt("What are its variants?", [], chat_history=history)

    assert "What is RAG?" in prompt
    assert "Retrieval-Augmented Generation" in prompt
    assert "What are its variants?" in prompt
