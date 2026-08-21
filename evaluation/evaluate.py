"""
evaluation/evaluate.py
=======================

Basic, reproducible evaluation of ResearchMate's retrieval and
generation quality against a small hand-labeled question set.

WHY EVALUATION MATTERS (AND WHY IT'S SEPARATE FROM THE APP)
------------------------------------------------------------------
A RAG demo that "looks like it works" in a couple of manual chat
tries can still be retrieving the wrong passages half the time, or
have the LLM ignoring the context and hallucinating. Without
measurement, you can't tell the difference between "the system is
actually good" and "I got lucky with the questions I tried."

This script answers two separate questions:

1. RETRIEVAL QUALITY (Recall@K): for each labeled question, did the
   retriever pull back at least one chunk from the EXPECTED source
   document, within the top-K results? This isolates retrieval
   correctness from generation quality — a bad retriever can't be
   rescued by a good LLM, so measuring this separately matters.

2. ANSWER QUALITY (basic correctness): does the generated answer
   contain enough keyword overlap with the expected answer to be
   plausibly correct? This is a crude, but transparent and
   reproducible, proxy — V1 deliberately avoids opaque "LLM-judges-LLM"
   scoring so you can see exactly why a question passed or failed.

HOW TO RUN
-------------
From the project root, with documents already processed into the
vector store (see README):

    python -m evaluation.evaluate

Results are printed to the console AND saved to
evaluation/results.json for later comparison across runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import TOP_K
from app.embeddings import EmbeddingManager
from app.generator import Generator
from app.retriever import Retriever
from app.vector_store import VectorStore

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

# Minimum fraction of expected-answer keywords that must appear in the
# generated answer for us to count it as "plausibly correct". This is
# intentionally simple and inspectable rather than a black-box metric.
KEYWORD_OVERLAP_THRESHOLD = 0.35

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "in", "on", "for",
    "that", "this", "with", "as", "by", "or", "be", "at", "from", "using",
    "used", "into", "its", "their", "which", "when", "each",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def evaluate() -> dict:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"No evaluation questions found at {QUESTIONS_PATH}")

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    vector_store = VectorStore()
    if vector_store.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Upload and process documents via the "
            "Streamlit app before running evaluation."
        )

    embedding_manager = EmbeddingManager()
    retriever = Retriever(vector_store, embedding_manager)
    generator = Generator()

    per_question_results = []
    retrieval_hits = 0
    answer_correct = 0

    for item in questions:
        question = item["question"]
        expected_source = item.get("source")
        expected_answer = item.get("expected_answer", "")

        retrieved = retriever.retrieve(question, top_k=TOP_K)
        retrieved_sources = {c.source for c in retrieved}

        retrieval_hit = expected_source in retrieved_sources if expected_source else None
        if retrieval_hit:
            retrieval_hits += 1

        gen_result = generator.generate_answer(question, retrieved, chat_history=None)

        expected_kw = _keywords(expected_answer)
        answer_kw = _keywords(gen_result.answer)
        overlap = len(expected_kw & answer_kw) / len(expected_kw) if expected_kw else None
        is_correct = (overlap is not None) and (overlap >= KEYWORD_OVERLAP_THRESHOLD)
        if is_correct:
            answer_correct += 1

        per_question_results.append(
            {
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": sorted(retrieved_sources),
                "retrieval_hit": retrieval_hit,
                "generated_answer": gen_result.answer,
                "expected_answer": expected_answer,
                "keyword_overlap": round(overlap, 3) if overlap is not None else None,
                "answer_correct": is_correct,
            }
        )

    n = len(questions)
    n_with_source = sum(1 for q in questions if q.get("source"))

    summary = {
        "num_questions": n,
        "retrieval_recall_at_k": round(retrieval_hits / n_with_source, 3) if n_with_source else None,
        "answer_correctness_rate": round(answer_correct / n, 3) if n else None,
        "num_questions_answered": n,  # all questions received a generated answer
        "top_k": TOP_K,
        "keyword_overlap_threshold": KEYWORD_OVERLAP_THRESHOLD,
    }

    output = {"summary": summary, "results": per_question_results}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


def _print_report(output: dict) -> None:
    summary = output["summary"]
    print("=" * 60)
    print("ResearchMate Evaluation Report")
    print("=" * 60)
    print(f"Questions evaluated:      {summary['num_questions']}")
    print(f"Retrieval Recall@{summary['top_k']}:      {summary['retrieval_recall_at_k']}")
    print(f"Answer correctness rate: {summary['answer_correctness_rate']}")
    print("-" * 60)
    for r in output["results"]:
        status = "✅" if r["answer_correct"] else "❌"
        hit = "✅" if r["retrieval_hit"] else "❌" if r["retrieval_hit"] is not None else "—"
        print(f"{status} (retrieval {hit}) {r['question']}")
    print("-" * 60)
    print(f"Full results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    report = evaluate()
    _print_report(report)
