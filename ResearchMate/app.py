"""
app.py
======

Streamlit UI for ResearchMate.

WHY app.py STAYS THIN
-------------------------
Every function called here (process_documents, Retriever.retrieve,
Generator.generate_answer, VectorStore.reset) is fully implemented in
app/*.py and testable in isolation, without Streamlit running.

This file's only job is:
- render widgets
- read user input
- call the pipeline
- render output
- measure performance

Keeping business logic OUT of app.py keeps the retrieval layer
independent from the UI.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st

from app.config import DEBUG_MODE, TOP_K
from app.embeddings import EmbeddingError, EmbeddingManager
from app.generator import Generator, GenerationError
from app.pipeline import process_documents
from app.retriever import Retriever
from app.vector_store import VectorStore, VectorStoreError


# ----------------------------------------------------------------------
# Streamlit configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="ResearchMate",
    page_icon="📄",
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached / session-persistent resources
# ----------------------------------------------------------------------

@st.cache_resource
def get_embedding_manager() -> EmbeddingManager:
    """
    Cached across reruns and sessions within this process.

    The embedding model is expensive to load, so we only load it once.
    """
    return EmbeddingManager()


@st.cache_resource
def get_vector_store() -> VectorStore:
    """
    Persistent ChromaDB connection.

    One VectorStore instance is reused by the Streamlit process.
    """
    return VectorStore()


def get_generator() -> Generator:
    """
    Creating a Generator is cheap because it only wraps the LLM provider.

    It is intentionally not cached so changes to the LLM_PROVIDER
    environment variable can take effect without restarting the app.
    """
    return Generator()


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize Streamlit session state variables."""

    defaults = {
        "uploaded_filenames": [],
        "pending_files": [],
        "chat_history": [],
        "last_sources": [],
        "last_debug": None,
        "debug_mode": DEBUG_MODE,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ----------------------------------------------------------------------
# Sidebar: document management
# ----------------------------------------------------------------------

with st.sidebar:

    st.header("📚 Documents")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploader",
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

    # --------------------------------------------------------------
    # Display uploaded files
    # --------------------------------------------------------------

    if st.session_state.uploaded_filenames:

        st.markdown("**Uploaded:**")

        for name in st.session_state.uploaded_filenames:
            st.markdown(f"✓ {name}")

    # --------------------------------------------------------------
    # Sidebar buttons
    # --------------------------------------------------------------

    process_clicked = st.button(
        "⚙️ Process Documents",
        use_container_width=True,
    )

    clear_clicked = st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------------------
    # Knowledge base statistics
    # --------------------------------------------------------------

    try:

        vs = get_vector_store()

        chunk_count = vs.count()
        doc_count = len(vs.get_indexed_document_ids())

        status = "Ready" if chunk_count > 0 else "Empty"

    except VectorStoreError as exc:

        chunk_count = 0
        doc_count = 0
        status = f"Error: {exc}"

    st.markdown("**Knowledge Base**")

    st.markdown(f"Documents: {doc_count}")

    st.markdown(f"Chunks: {chunk_count:,}")

    st.markdown(f"Status: {status}")

    st.divider()

    # --------------------------------------------------------------
    # Debug mode
    # --------------------------------------------------------------

    st.session_state.debug_mode = st.checkbox(
        "🔍 Debug mode",
        value=st.session_state.debug_mode,
        help=(
            "Show the retrieved chunks, similarity scores, "
            "full prompt sent to the LLM, and performance timings."
        ),
    )


# ----------------------------------------------------------------------
# Handle: Clear Knowledge Base
# ----------------------------------------------------------------------

if clear_clicked:

    try:

        get_vector_store().reset()

        st.session_state.uploaded_filenames = []
        st.session_state.pending_files = []
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        st.session_state.last_debug = None

        st.sidebar.success(
            "Knowledge base cleared."
        )

    except VectorStoreError as exc:

        st.sidebar.error(
            f"Couldn't clear the knowledge base: {exc}"
        )


# ----------------------------------------------------------------------
# Handle: Process Documents
# ----------------------------------------------------------------------

if process_clicked:

    if not st.session_state.pending_files:

        st.sidebar.warning(
            "Upload at least one PDF first."
        )

    else:

        with st.spinner(
            "Processing documents "
            "(extracting text, chunking, embedding)..."
        ):

            try:

                report = process_documents(
                    st.session_state.pending_files,
                    vector_store=get_vector_store(),
                    embedding_manager=get_embedding_manager(),
                )

                # --------------------------------------------------
                # Successfully processed files
                # --------------------------------------------------

                if report.files_processed:

                    st.sidebar.success(
                        f"Indexed "
                        f"{len(report.files_processed)} file(s), "
                        f"{report.chunks_added} chunk(s) added."
                    )

                # --------------------------------------------------
                # Skipped files
                # --------------------------------------------------

                if report.files_skipped:

                    st.sidebar.info(
                        "Skipped (already indexed): "
                        + ", ".join(report.files_skipped)
                    )

                # --------------------------------------------------
                # Failed files
                # --------------------------------------------------

                for fname, err in report.files_failed.items():

                    st.sidebar.error(
                        f"{fname}: {err}"
                    )

            except EmbeddingError as exc:

                st.sidebar.error(
                    f"Embedding failed: {exc}"
                )

            except VectorStoreError as exc:

                st.sidebar.error(
                    f"Vector store failed: {exc}"
                )


# ----------------------------------------------------------------------
# Main area: Chat interface
# ----------------------------------------------------------------------

st.title("ResearchMate")

st.caption(
    "Your AI research assistant for understanding academic papers."
)

st.subheader("💬 Ask ResearchMate")


query = st.text_input(
    "Ask a question about your uploaded papers",
    key="query_input",
)

ask_clicked = st.button("Ask")


# ----------------------------------------------------------------------
# Handle user question
# ----------------------------------------------------------------------

if ask_clicked:

    # --------------------------------------------------------------
    # Empty question
    # --------------------------------------------------------------

    if not query or not query.strip():

        st.warning(
            "Please enter a question."
        )

    # --------------------------------------------------------------
    # No documents
    # --------------------------------------------------------------

    elif get_vector_store().count() == 0:

        st.warning(
            "No documents indexed yet. "
            "Upload and process a PDF first."
        )

    # --------------------------------------------------------------
    # Process question
    # --------------------------------------------------------------

    else:

        with st.spinner(
            "Retrieving relevant passages and generating an answer..."
        ):

            try:

                # ==================================================
                # START TOTAL TIMER
                # ==================================================

                total_start = time.perf_counter()

                # ==================================================
                # 1. RETRIEVAL
                # ==================================================

                retrieval_start = time.perf_counter()

                retriever = Retriever(
                    get_vector_store(),
                    get_embedding_manager(),
                )

                retrieved = retriever.retrieve(
                    query,
                    top_k=TOP_K,
                )

                retrieval_time = (
                    time.perf_counter()
                    - retrieval_start
                )

                # ==================================================
                # 2. LLM GENERATION
                # ==================================================

                generation_start = time.perf_counter()

                generator = get_generator()

                result = generator.generate_answer(
                    query,
                    retrieved,
                    chat_history=st.session_state.chat_history,
                )

                generation_time = (
                    time.perf_counter()
                    - generation_start
                )

                # ==================================================
                # 3. TOTAL RESPONSE TIME
                # ==================================================

                total_time = (
                    time.perf_counter()
                    - total_start
                )

                # ==================================================
                # PRINT PERFORMANCE TO TERMINAL
                # ==================================================

                print()
                print("=" * 55)
                print("ResearchMate Performance")
                print("=" * 55)

                print(
                    f"Retrieval time:   "
                    f"{retrieval_time:.2f} seconds"
                )

                print(
                    f"Generation time:  "
                    f"{generation_time:.2f} seconds"
                )

                print(
                    f"Total time:       "
                    f"{total_time:.2f} seconds"
                )

                print(
                    f"Chunks retrieved: "
                    f"{len(retrieved)}"
                )

                print("=" * 55)
                print()

                # ==================================================
                # SAVE CHAT HISTORY
                # ==================================================

                st.session_state.chat_history.append(
                    (
                        query,
                        result.answer,
                    )
                )

                # ==================================================
                # SAVE SOURCES
                # ==================================================

                st.session_state.last_sources = result.sources

                # ==================================================
                # SAVE DEBUG INFORMATION
                # ==================================================

                if st.session_state.debug_mode:

                    st.session_state.last_debug = {
                        "query": query,
                        "retrieved": retrieved,
                        "prompt": result.prompt_used,
                        "retrieval_time": retrieval_time,
                        "generation_time": generation_time,
                        "total_time": total_time,
                    }

            # ------------------------------------------------------
            # Error handling
            # ------------------------------------------------------

            except EmbeddingError as exc:

                st.error(
                    f"Couldn't embed your query: {exc}"
                )

            except VectorStoreError as exc:

                st.error(
                    f"Vector store error: {exc}"
                )

            except GenerationError as exc:

                st.error(
                    f"The LLM couldn't generate an answer: {exc}"
                )


# ----------------------------------------------------------------------
# Render conversation history
# ----------------------------------------------------------------------

for i, (user_msg, assistant_msg) in enumerate(
    st.session_state.chat_history
):

    st.markdown(
        f"**You:** {user_msg}"
    )

    st.markdown("**Answer:**")

    st.markdown(assistant_msg)

    # --------------------------------------------------------------
    # Show sources only under the latest answer
    # --------------------------------------------------------------

    if (
        i == len(st.session_state.chat_history) - 1
        and st.session_state.last_sources
    ):

        st.markdown("**📚 Sources**")

        for chunk in st.session_state.last_sources:

            with st.expander(
                f"📄 {chunk.source} — "
                f"Page {chunk.page} "
                f"(similarity: {chunk.similarity:.2f})"
            ):

                st.write(chunk.text)

    st.divider()


# ----------------------------------------------------------------------
# Debug panel
# ----------------------------------------------------------------------

if (
    st.session_state.debug_mode
    and st.session_state.last_debug
):

    with st.expander(
        "🔍 Debug: retrieval + prompt + performance",
        expanded=True,
    ):

        debug = st.session_state.last_debug

        # ----------------------------------------------------------
        # Query
        # ----------------------------------------------------------

        st.markdown(
            f"**Query:** {debug['query']}"
        )

        # ----------------------------------------------------------
        # Performance
        # ----------------------------------------------------------

        st.markdown("### ⚡ Performance")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Retrieval",
                f"{debug['retrieval_time']:.2f}s",
            )

        with col2:

            st.metric(
                "LLM Generation",
                f"{debug['generation_time']:.2f}s",
            )

        with col3:

            st.metric(
                "Total",
                f"{debug['total_time']:.2f}s",
            )

        st.markdown(
            f"**Chunks retrieved:** "
            f"{len(debug['retrieved'])}"
        )

        # ----------------------------------------------------------
        # Retrieved chunks
        # ----------------------------------------------------------

        st.markdown(
            "### 📚 Retrieved chunks"
        )

        for i, chunk in enumerate(
            debug["retrieved"],
            start=1,
        ):

            st.markdown(
                f"{i}. `{chunk.source}` "
                f"p.{chunk.page} — "
                f"distance={chunk.distance:.4f}, "
                f"similarity={chunk.similarity:.2f}"
            )

        # ----------------------------------------------------------
        # Full prompt
        # ----------------------------------------------------------

        st.markdown(
            "### 🧠 Full prompt sent to the LLM"
        )

        st.code(
            debug["prompt"],
            language="text",
        )