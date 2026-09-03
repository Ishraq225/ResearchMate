"""
config.py
=========

Centralized configuration for ResearchMate.

WHY THIS FILE EXISTS
---------------------
Without a config module, values like `chunk_size=800` or the embedding
model name end up copy-pasted across ingestion.py, chunking.py,
embeddings.py, etc. When you later want to experiment (e.g. try
chunk_size=500), you'd have to hunt through every file.

By centralizing config:
1. Every module imports the SAME values -> no drift between them.
2. You can override anything via environment variables (.env) without
   touching code -> good practice for secrets and per-environment settings.
3. It documents, in one place, every "knob" this RAG system has.

WHAT WOULD HAPPEN WITHOUT IT
------------------------------
Silent bugs: e.g. chunking.py uses chunk_size=800 but embeddings.py
assumes 500-token chunks when computing batch sizes. Changing one
value in one file would produce inconsistent behavior across the
pipeline with no error message telling you why retrieval got worse.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a .env file (if present) into the environment.
# This must happen before we read any os.environ values below.
load_dotenv()

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # .../ResearchMate/
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"          # raw uploaded PDFs
VECTOR_STORE_DIR = DATA_DIR / "vector_store"    # persisted ChromaDB

# Make sure these directories exist at import time so the rest of the
# app can assume they're always there.
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# CHUNKING
# ------------------------------------------------------------------
# chunk_size: how many characters go into one chunk.
#   Too small -> chunks lack enough context for the LLM to reason with.
#   Too large -> retrieval gets less precise (a chunk about 3 subtopics
#   might get pulled in for a query about only one of them), and you
#   waste context-window space on irrelevant text.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))

# chunk_overlap: how many characters consecutive chunks share.
#   Without overlap, a sentence that explains a key idea could be
#   split exactly at the chunk boundary, so neither chunk contains the
#   full idea and retrieval can miss it entirely. Overlap acts as a
#   safety margin so ideas near chunk edges still appear whole in at
#   least one chunk.
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# ------------------------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------------------------
# all-MiniLM-L6-v2: small (~80MB), fast on CPU, good enough quality
# for a portfolio project. Swapping this later only requires changing
# this one line, because embeddings.py will read from here.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# ------------------------------------------------------------------
# VECTOR STORE
# ------------------------------------------------------------------
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "researchmate_docs")

# ------------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------------
# How many chunks to pull back per query. Configurable because it's
# a direct trade-off: higher = more recall (less likely to miss the
# right passage) but more noise in the prompt and higher LLM cost.
TOP_K = int(os.getenv("TOP_K", 5))

# ------------------------------------------------------------------
# LLM PROVIDER (kept modular — see generator.py in a later phase)
# ------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ------------------------------------------------------------------
# DEBUG MODE
# ------------------------------------------------------------------
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# ------------------------------------------------------------------
# V2: CONVERSATION MEMORY
# ------------------------------------------------------------------
# How many recent (user, assistant) turns to include in the prompt.
# See app/memory.py for why a fixed window (not summarization) is used.
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", 6))

# ------------------------------------------------------------------
# V2: INSUFFICIENT EVIDENCE THRESHOLD
# ------------------------------------------------------------------
# Minimum similarity (0-1, see RetrievedChunk.similarity) that the BEST
# retrieved chunk must reach before we trust the LLM to answer from it.
# Below this, we short-circuit with an explicit "insufficient evidence"
# response rather than risk the LLM leaning on its own pretrained
# knowledge for a low-relevance retrieval. Tuned conservatively low —
# it only catches genuinely weak matches, not just "not the top hit".
MIN_EVIDENCE_SIMILARITY = float(os.getenv("MIN_EVIDENCE_SIMILARITY", 0.2))
