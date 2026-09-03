"""
research_modes.py
==================

V2 "Research Modes": Ask, Summarize, Compare, Explain, Find Evidence.

WHY MODES ARE A SEPARATE MODULE (NOT MORE if/else IN generator.py)
------------------------------------------------------------------------
Each mode needs its own system prompt (different instructions to the
LLM) and, in some cases, a different framing of the user-turn prompt
(e.g. Summarize doesn't really have a "question" — it has a target
document set). Putting all five as a growing if/else chain inside
generator.py would make that file harder to read and harder to test
mode-by-mode in isolation. Instead, this module owns "what does each
mode need from the LLM", and generator.py just asks it for a
(system_prompt, user_prompt) pair — it doesn't know or care about mode
internals.

WHY THESE FIVE MODES SPECIFICALLY
--------------------------------------
Each corresponds to a genuinely different task shape a researcher
performs — free-form Q&A, structured summarization, side-by-side
comparison, concept explanation, and evidence-finding for/against a
claim. Modeling them as distinct prompt strategies (rather than one
mega-prompt trying to do everything) keeps each mode's output
predictable and easy to reason about.
"""

from __future__ import annotations

from enum import Enum

BASE_RULES = """You are ResearchMate, an AI research assistant that helps users \
understand and compare academic papers.

Ground rules that apply in every mode:
1. Use the retrieved context as your primary and preferred evidence.
2. Do not invent facts, page numbers, or citations that are not supported by the \
retrieved context.
3. Do not claim a paper says something unless the retrieved context actually \
supports it.
4. When multiple documents are involved, clearly attribute which claim comes from \
which paper.
5. If the retrieved context is insufficient to answer confidently, say so explicitly \
instead of relying on your own general knowledge. Evidence from the documents always \
takes priority over what you already know about the topic.
6. Cite the relevant source document (and page, if useful) after important claims, \
using the format [source: FILENAME, page N].
"""


class ResearchMode(str, Enum):
    ASK = "Ask"
    SUMMARIZE = "Summarize"
    COMPARE = "Compare"
    EXPLAIN = "Explain"
    FIND_EVIDENCE = "Find Evidence"


MODE_SYSTEM_PROMPTS: dict[ResearchMode, str] = {
    ResearchMode.ASK: BASE_RULES + """
Mode: ASK — answer the user's question directly and concisely, using only the \
provided context.
""",
    ResearchMode.SUMMARIZE: BASE_RULES + """
Mode: SUMMARIZE — produce a structured summary of the retrieved context using \
exactly these sections, skipping a section only if the context truly has nothing \
relevant to it:

Overview
Key Contributions
Methodology
Results
Limitations
Conclusion
""",
    ResearchMode.COMPARE: BASE_RULES + """
Mode: COMPARE — compare the documents/approaches referenced in the retrieved \
context. Structure your answer as similarities and differences, and explicitly name \
which source each point comes from. Do not compare documents that were not retrieved \
in the context — if only one document's evidence was retrieved, say a comparison \
isn't possible from the current evidence and explain what's missing.
""",
    ResearchMode.EXPLAIN: BASE_RULES + """
Mode: EXPLAIN — explain the requested concept clearly, as if teaching a curious \
beginner, while staying grounded in the retrieved context. Prefer plain language over \
jargon, and introduce necessary technical terms with a short definition the first \
time they appear.
""",
    ResearchMode.FIND_EVIDENCE: BASE_RULES + """
Mode: FIND EVIDENCE — the user has stated a claim. Search the retrieved context for \
passages that SUPPORT the claim and passages that CONTRADICT or complicate it. \
Present both explicitly under "Supporting evidence" and "Contradicting or \
complicating evidence" headings, citing sources for each. If no evidence either way \
is found in the retrieved context, say so plainly rather than guessing.
""",
}


def get_system_prompt(mode: ResearchMode | str) -> str:
    """Look up the system prompt for a mode, tolerating a plain string mode name."""
    if isinstance(mode, str):
        mode = ResearchMode(mode)
    return MODE_SYSTEM_PROMPTS[mode]


def build_user_instruction(mode: ResearchMode | str, query: str) -> str:
    """
    Frame the user's raw input according to the mode. Ask/Explain/
    FindEvidence use the query as-is (it's already phrased as a
    question or claim); Summarize/Compare rephrase an empty/short
    query into a sensible default instruction so the user isn't forced
    to type "summarize this" every time they switch modes.
    """
    if isinstance(mode, str):
        mode = ResearchMode(mode)

    query = (query or "").strip()

    if mode == ResearchMode.SUMMARIZE and not query:
        return "Summarize the selected document(s)."
    if mode == ResearchMode.COMPARE and not query:
        return "Compare the selected documents."
    return query
