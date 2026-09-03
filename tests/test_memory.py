"""
Tests for app/memory.py (ConversationMemory).
"""

from app.memory import ConversationMemory


def test_follow_up_question_retains_context():
    memory = ConversationMemory()
    memory.add_turn("What is RAG?", "RAG stands for Retrieval-Augmented Generation.")

    history = memory.recent_history()

    assert history == [("What is RAG?", "RAG stands for Retrieval-Augmented Generation.")]


def test_new_chat_clears_history():
    memory = ConversationMemory()
    memory.add_turn("What is RAG?", "RAG is...")
    memory.add_turn("What are its variants?", "RAG-Sequence and RAG-Token.")

    memory.new_chat()

    assert memory.recent_history() == []
    assert memory.is_empty()


def test_clear_conversation_button_behaves_like_new_chat():
    memory = ConversationMemory()
    memory.add_turn("Q", "A")
    memory.clear()
    assert memory.is_empty()


def test_history_window_is_bounded():
    memory = ConversationMemory(max_turns=2)
    memory.add_turn("Q1", "A1")
    memory.add_turn("Q2", "A2")
    memory.add_turn("Q3", "A3")

    history = memory.recent_history()

    # Only the 2 most recent turns should be included, oldest first.
    assert history == [("Q2", "A2"), ("Q3", "A3")]


def test_full_turn_history_is_preserved_for_ui_even_when_window_is_bounded():
    memory = ConversationMemory(max_turns=1)
    memory.add_turn("Q1", "A1")
    memory.add_turn("Q2", "A2")

    # The prompt window is bounded, but the UI needs the FULL list to
    # replay the conversation, so `turns` itself must not be truncated.
    assert len(memory.turns) == 2
    assert memory.recent_history() == [("Q2", "A2")]
