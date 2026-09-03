"""
memory.py
=========

Conversation memory: lets ResearchMate understand follow-up questions
("What are its advantages?") by carrying recent turns into the prompt.

WHY THIS IS ITS OWN MODULE
-------------------------------
In V1, chat history was just a raw Python list passed straight into
`build_rag_prompt`. That worked for a short demo but has two real
problems as conversations grow:

1. UNBOUNDED GROWTH: every turn added to the prompt forever. A 30-turn
   conversation would eventually make every subsequent request slower,
   more expensive, and risk exceeding the LLM's context window.
2. NO SEPARATION OF CONCERNS: "how much history to include" was
   decided inline wherever the prompt was built, making it hard to
   change the windowing strategy in one place.

ConversationMemory solves both: it owns the history list, decides how
many recent turns are "relevant enough" to include (a simple, clean,
inspectable windowing strategy — not summarization, which V2
deliberately avoids to keep behavior transparent), and exposes
`new_chat()` / `clear()` as explicit, intentional actions matching the
"New Chat" / "Clear Conversation" buttons in the sidebar.

WHY A SIMPLE WINDOW INSTEAD OF SUMMARIZATION
-------------------------------------------------
Summarizing history with another LLM call would add latency, cost, and
a second place where hallucination could creep in ("did the summary
correctly preserve what was asked?"). A fixed window of the N most
recent turns is transparent, free, and sufficient for the kind of
short-range pronoun resolution ("its", "that paper") this project
targets. Deep multi-session memory is out of scope for V2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAX_TURNS = 6  # how many recent (user, assistant) turns to keep in the prompt


@dataclass
class Turn:
    """One exchange in the conversation."""

    user_message: str
    assistant_message: str
    sources: list = field(default_factory=list)   # RetrievedChunk list, for UI replay
    mode: str = "Ask"                                # research mode active for this turn


class ConversationMemory:
    """
    Holds the full conversation history (for UI display) and exposes a
    bounded "recent window" for prompt construction.
    """

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        self.max_turns = max_turns
        self.turns: list[Turn] = []

    def add_turn(self, user_message: str, assistant_message: str, sources=None, mode: str = "Ask") -> None:
        self.turns.append(
            Turn(user_message=user_message, assistant_message=assistant_message, sources=sources or [], mode=mode)
        )

    def recent_history(self) -> list[tuple[str, str]]:
        """
        Returns up to `max_turns` most recent (user, assistant) pairs,
        oldest first — the shape `build_rag_prompt` expects.

        Keeping this windowed (rather than the full history) is what
        satisfies the "do not allow conversation history to grow
        indefinitely" requirement.
        """
        windowed = self.turns[-self.max_turns:]
        return [(t.user_message, t.assistant_message) for t in windowed]

    def new_chat(self) -> None:
        """Explicit action matching the '[New Chat]' button — same as clear()."""
        self.turns = []

    def clear(self) -> None:
        """Explicit action matching the '[Clear Conversation]' button."""
        self.turns = []

    def is_empty(self) -> bool:
        return len(self.turns) == 0
