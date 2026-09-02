"""
prompts.py
==========

Central store of prompt text for ResearchMate.

WHY PROMPTS LIVE IN THEIR OWN FILE
--------------------------------------
Prompt wording is something you iterate on constantly while building
a RAG system — tightening instructions, fixing hallucination issues,
adjusting tone. If prompt strings were scattered inline inside
generator.py's function calls, every tweak would mean digging through
logic code. Keeping them here means:
1. You can review/tune the actual instructions to the LLM at a glance.
2. generator.py stays focused on orchestration logic, not string
   templates.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are ResearchMate, an AI research assistant that helps users \
understand academic papers.

Answer the user's question using ONLY the provided research-paper context below.

Rules:
1. Do not fabricate facts or invent information that is not present in the context.
2. If the context does not contain enough information to answer, say so explicitly \
— do not guess.
3. Explain technical concepts clearly and concisely.
4. Distinguish between what is directly stated in the sources and any reasonable \
inference you make to connect ideas — flag inferences explicitly.
5. Cite the relevant source document (and page, if useful) after important claims, \
using the format [source: FILENAME, page N].
6. Never claim information is present in the papers unless it is actually supported \
by the retrieved context shown to you.
"""


def build_rag_prompt(
    query: str,
    context_chunks: list[dict],
    chat_history: list[tuple[str, str]] | None = None,
) -> str:
    """
    Assemble the full user-turn prompt sent to the LLM: conversation
    history (if any) + retrieved context + the current question.

    Args:
        query: the user's current question.
        context_chunks: list of dicts like
            {"text": ..., "source": ..., "page": ..., "chunk_id": ...}
            — these come from retriever.RetrievedChunk, converted to
            plain dicts by generator.py.
        chat_history: list of (user_message, assistant_message) tuples
            from earlier turns in the conversation, oldest first.

    Returns:
        A single string ready to send as the user-role message content.
    """
    parts: list[str] = []

    if chat_history:
        parts.append("Conversation so far (for context on pronouns/follow-ups):")
        for user_msg, assistant_msg in chat_history:
            parts.append(f"User: {user_msg}")
            parts.append(f"Assistant: {assistant_msg}")
        parts.append("")  # blank line separator

    parts.append("Retrieved context from the uploaded papers:")
    parts.append("")
    if not context_chunks:
        parts.append("(No relevant context was found in the uploaded documents.)")
    else:
        for i, chunk in enumerate(context_chunks, start=1):
            parts.append(
                f"[Chunk {i} | source: {chunk['source']}, page: {chunk['page']}]\n{chunk['text']}"
            )
            parts.append("")

    parts.append(f"Question: {query}")
    parts.append("")
    parts.append(
        "Answer the question using only the context above. If the context is "
        "insufficient, say so clearly instead of guessing."
    )

    return "\n".join(parts)
