"""
app.py
======

Streamlit UI for ResearchMate V2.

WHY app.py STAYS THIN (unchanged principle from V1)
--------------------------------------------------------
Every meaningful decision (which documents to search, how much
history to include, whether evidence is sufficient) is made inside
app/*.py modules and is independently testable. This file only wires
widgets to those modules and renders results — it still contains no
retrieval or generation logic of its own.

WHAT CHANGED FROM V1
-------------------------
- Sidebar now lists documents individually with checkboxes (document
  selection/filtering — V2 feature) instead of just a flat name list.
- A research-mode selector (Ask/Summarize/Compare/Explain/Find
  Evidence) drives which prompt strategy generator.py uses.
- Conversation memory is now owned by app.memory.ConversationMemory
  instead of a raw list in session_state, with explicit "New Chat" /
  "Clear Conversation" actions.
- Chat rendered with st.chat_message / st.chat_input instead of a
  manual text_input + button, per the V2 UI requirements.
- Insufficient-evidence responses are visually distinguished so users
  don't mistake "no evidence found" for a real grounded answer.
"""

from __future__ import annotations

import streamlit as st

from app.config import DEBUG_MODE, TOP_K
from app.embeddings import EmbeddingError, EmbeddingManager
from app.generator import Generator, GenerationError
from app.memory import ConversationMemory
from app.pipeline import process_documents
from app.research_modes import ResearchMode
from app.retriever import Retriever
from app.vector_store import VectorStore, VectorStoreError

st.set_page_config(page_title="ResearchMate", page_icon="📄", layout="wide")


# ----------------------------------------------------------------------
# Cached / session-persistent resources
# ----------------------------------------------------------------------
@st.cache_resource
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()


@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore()


def get_generator() -> Generator:
    """Not cached: cheap to construct, and lets LLM_PROVIDER changes take effect live."""
    return Generator()


def init_session_state() -> None:
    defaults = {
        "uploaded_filenames": [],
        "pending_files": [],
        "memory": ConversationMemory(),
        "selected_document_ids": set(),  # empty set = "no documents selected"
        "debug_mode": DEBUG_MODE,
        "research_mode": ResearchMode.ASK.value,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ----------------------------------------------------------------------
# Sidebar: document management + selection + conversation controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Documents")

    uploaded = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True, key="uploader"
    )

    if uploaded:
        from app.config import DOCUMENTS_DIR

        saved_paths = []
        for f in uploaded:
            dest = DOCUMENTS_DIR / f.name
            if not dest.exists():
                dest.write_bytes(f.getbuffer())
            saved_paths.append(dest)
            if f.name not in st.session_state.uploaded_filenames:
                st.session_state.uploaded_filenames.append(f.name)
        st.session_state.pending_files = saved_paths

    process_clicked = st.button("⚙️ Process Documents", use_container_width=True)
    clear_kb_clicked = st.button("🗑️ Clear Knowledge Base", use_container_width=True)

    st.divider()

    # --- Document selection (V2): one checkbox per indexed document ---
    try:
        vs = get_vector_store()
        registry = vs.get_document_registry()  # {document_id: filename}
    except VectorStoreError as exc:
        registry = {}
        st.error(f"Couldn't load document list: {exc}")

    if registry:
        st.markdown("**Selected Sources**")
        if "selected_document_ids" not in st.session_state or not st.session_state.selected_document_ids:
            # Default: everything selected on first load so new users
            # get sensible behavior without manually ticking boxes.
            st.session_state.selected_document_ids = set(registry.keys())

        for doc_id, filename in sorted(registry.items(), key=lambda kv: kv[1].lower()):
            checked = st.checkbox(
                filename, value=doc_id in st.session_state.selected_document_ids, key=f"doc_select_{doc_id}"
            )
            if checked:
                st.session_state.selected_document_ids.add(doc_id)
            else:
                st.session_state.selected_document_ids.discard(doc_id)
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # --- Knowledge base stats ---
    try:
        chunk_count = vs.count()
        doc_count = len(registry)
        status = "Ready" if chunk_count > 0 else "Empty"
    except VectorStoreError as exc:
        chunk_count, doc_count, status = 0, 0, f"Error: {exc}"

    st.markdown("**Knowledge Base**")
    col1, col2 = st.columns(2)
    col1.metric("Documents", doc_count)
    col2.metric("Chunks", chunk_count)
    st.caption(f"Status: {status}")

    st.divider()

    st.markdown("**Chat**")
    new_chat_clicked = st.button("🆕 New Chat", use_container_width=True)
    clear_chat_clicked = st.button("🧹 Clear Conversation", use_container_width=True)

    st.divider()
    st.session_state.debug_mode = st.checkbox(
        "🔍 Debug mode", value=st.session_state.debug_mode,
        help="Show retrieved chunks, similarity scores, and the full prompt sent to the LLM.",
    )


# ----------------------------------------------------------------------
# Handle sidebar button actions
# ----------------------------------------------------------------------
if clear_kb_clicked:
    try:
        get_vector_store().reset()
        st.session_state.uploaded_filenames = []
        st.session_state.pending_files = []
        st.session_state.selected_document_ids = set()
        st.session_state.memory.clear()
        st.sidebar.success("Knowledge base cleared.")
    except VectorStoreError as exc:
        st.sidebar.error(f"Couldn't clear the knowledge base: {exc}")

if new_chat_clicked or clear_chat_clicked:
    st.session_state.memory.new_chat()
    st.sidebar.success("Conversation cleared.")

if process_clicked:
    if not st.session_state.pending_files:
        st.sidebar.warning("Upload at least one PDF first.")
    else:
        with st.spinner("Processing documents (extracting text, chunking, embedding)..."):
            try:
                report = process_documents(
                    st.session_state.pending_files,
                    vector_store=get_vector_store(),
                    embedding_manager=get_embedding_manager(),
                )
                if report.files_processed:
                    st.sidebar.success(f"Indexed {len(report.files_processed)} new file(s).")
                if report.files_reprocessed:
                    st.sidebar.info(
                        f"Re-indexed (content changed): {', '.join(report.files_reprocessed)}"
                    )
                if report.files_skipped:
                    st.sidebar.info(f"Skipped (unchanged): {', '.join(report.files_skipped)}")
                for fname, err in report.files_failed.items():
                    st.sidebar.error(f"{fname}: {err}")
                # Newly indexed documents should be selected by default.
                st.session_state.selected_document_ids = set(get_vector_store().get_document_registry().keys())
            except EmbeddingError as exc:
                st.sidebar.error(f"Embedding failed: {exc}")
            except VectorStoreError as exc:
                st.sidebar.error(f"Vector store failed: {exc}")


# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.title("ResearchMate")
st.caption("Your AI research assistant for understanding academic papers.")

mode_tabs = [m.value for m in ResearchMode]
st.session_state.research_mode = st.radio(
    "Research Mode", mode_tabs, horizontal=True,
    index=mode_tabs.index(st.session_state.research_mode),
)

# --- Render conversation history using chat bubbles ---
for turn in st.session_state.memory.turns:
    with st.chat_message("user"):
        st.markdown(turn.user_message)
    with st.chat_message("assistant"):
        st.markdown(turn.assistant_message)
        if turn.sources:
            with st.expander("📚 Sources"):
                for chunk in turn.sources:
                    st.markdown(
                        f"📄 **{chunk.source}** — Page {chunk.page} "
                        f"(similarity: {chunk.similarity:.2f})"
                    )
                    st.caption(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))

# --- Chat input ---
placeholder = {
    ResearchMode.ASK.value: "Ask a question about your selected papers...",
    ResearchMode.SUMMARIZE.value: "Press enter to summarize the selected documents (or add focus instructions)...",
    ResearchMode.COMPARE.value: "What would you like to compare across the selected papers?",
    ResearchMode.EXPLAIN.value: "What concept would you like explained?",
    ResearchMode.FIND_EVIDENCE.value: "State the claim you want evidence for or against...",
}[st.session_state.research_mode]

user_input = st.chat_input(placeholder)

if user_input is not None:
    query = user_input.strip()
    allow_empty = st.session_state.research_mode in (ResearchMode.SUMMARIZE.value, ResearchMode.COMPARE.value)

    if not query and not allow_empty:
        st.warning("Please enter a question.")
    elif get_vector_store().count() == 0:
        st.warning("No documents indexed yet. Upload and process a PDF first.")
    elif not st.session_state.selected_document_ids:
        st.warning("No documents selected. Check at least one document in the sidebar.")
    else:
        with st.chat_message("user"):
            st.markdown(query if query else f"({st.session_state.research_mode} — no specific query)")

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and generating an answer..."):
                try:
                    retriever = Retriever(get_vector_store(), get_embedding_manager())
                    retrieved = retriever.retrieve(
                        query or st.session_state.research_mode,
                        top_k=TOP_K,
                        document_ids=list(st.session_state.selected_document_ids),
                    )

                    generator = get_generator()
                    result = generator.generate_answer(
                        query,
                        retrieved,
                        chat_history=st.session_state.memory.recent_history(),
                        mode=st.session_state.research_mode,
                    )

                    if result.insufficient_evidence:
                        st.warning(result.answer)
                    else:
                        st.markdown(result.answer)

                    if result.sources:
                        with st.expander("📚 Sources", expanded=False):
                            for chunk in result.sources:
                                st.markdown(
                                    f"📄 **{chunk.source}** — Page {chunk.page} "
                                    f"(similarity: {chunk.similarity:.2f})"
                                )
                                st.caption(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))

                    st.session_state.memory.add_turn(
                        query, result.answer, sources=result.sources, mode=st.session_state.research_mode
                    )

                    if st.session_state.debug_mode:
                        with st.expander("🔍 Debug: retrieval + prompt details", expanded=True):
                            st.markdown(f"**Mode:** {st.session_state.research_mode}")
                            st.markdown(f"**Documents searched:** {len(st.session_state.selected_document_ids)}")
                            st.markdown("**Retrieved chunks (ranked):**")
                            for i, chunk in enumerate(retrieved, start=1):
                                st.markdown(
                                    f"{i}. `{chunk.source}` p.{chunk.page} — "
                                    f"distance={chunk.distance:.4f}, similarity={chunk.similarity:.2f}"
                                )
                            st.markdown("**System prompt:**")
                            st.code(result.system_prompt_used, language="text")
                            st.markdown("**Full prompt sent to the LLM:**")
                            st.code(result.prompt_used, language="text")

                except EmbeddingError as exc:
                    st.error(f"Couldn't embed your query: {exc}")
                except VectorStoreError as exc:
                    st.error(f"Vector store error: {exc}")
                except GenerationError as exc:
                    st.error(f"The LLM couldn't generate an answer: {exc}")
