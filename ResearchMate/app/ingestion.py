"""
ingestion.py
============

Responsible for ONE thing: turning uploaded PDF files into raw text,
page by page, while preserving metadata needed for later citations.

WHY THIS STAGE EXISTS
----------------------
An LLM can't read a PDF file directly. We need to convert the binary
PDF into plain text before anything else (chunking, embedding, etc.)
can happen. This module is intentionally "dumb" — it does not chunk,
embed, or interpret the text. Keeping it narrow makes it easy to test
and easy to swap out (e.g. add OCR support later) without touching
any other stage of the pipeline.

WHAT WOULD HAPPEN WITHOUT CAREFUL METADATA HANDLING
-----------------------------------------------------
If we only kept the extracted text and threw away which file/page it
came from, we would be able to generate an answer but NEVER be able
to tell the user "this claim came from RAG.pdf, page 4." Source
attribution (a core requirement of this project) starts here.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class IngestionError(Exception):
    """Raised when a PDF cannot be read or contains no usable text."""


@dataclass
class PageDocument:
    """
    Represents the raw text extracted from a single PDF page.

    This is the atomic unit that chunking.py will later split further.
    We keep one PageDocument per page (not per whole PDF) because page
    numbers are part of what we cite to the user.
    """

    text: str
    source: str          # original filename, e.g. "RAG.pdf"
    page: int             # 1-indexed page number
    document_id: str      # stable slug derived from filename
    metadata: dict = field(default_factory=dict)


def _make_document_id(filename: str) -> str:
    """
    Turn 'Retrieval Augmented Generation.pdf' into 'retrieval_augmented_generation_pdf'.

    A stable, filesystem/collection-safe ID is needed because:
    - ChromaDB metadata filtering works better with clean string IDs.
    - We use this ID to detect "have I already indexed this file?"
      (see vector_store.py) so we don't recompute embeddings for
      documents that haven't changed.
    """
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug or f"doc_{uuid.uuid4().hex[:8]}"


def load_pdf(file_path: str | Path, original_filename: str | None = None) -> list[PageDocument]:
    """
    Load a single PDF file and extract text from every page.

    Args:
        file_path: path to the PDF on disk.
        original_filename: the filename to record as the citation
            source. Defaults to file_path's name. Useful when the
            file is stored under a temp name but should be cited
            using its original upload name.

    Returns:
        A list of PageDocument, one per non-empty page.

    Raises:
        IngestionError: if the file can't be parsed, isn't a valid
            PDF, or contains no extractable text at all (e.g. a
            scanned/image-only PDF with no OCR layer — V1 does not
            support OCR, see project scope).
    """
    file_path = Path(file_path)
    filename = original_filename or file_path.name

    if not file_path.exists():
        raise IngestionError(f"File not found: {file_path}")

    try:
        reader = PdfReader(str(file_path))
    except PdfReadError as exc:
        raise IngestionError(f"'{filename}' is not a valid or readable PDF.") from exc
    except Exception as exc:  # noqa: BLE001 - we want to convert ANY read failure
        raise IngestionError(f"Failed to open '{filename}': {exc}") from exc

    if reader.is_encrypted:
        # We don't ask for passwords in V1 — fail clearly instead of
        # silently returning empty text.
        raise IngestionError(
            f"'{filename}' is password-protected. Remove the password and re-upload."
        )

    document_id = _make_document_id(filename)
    pages: list[PageDocument] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - a single bad page shouldn't kill the whole PDF
            raw_text = ""

        cleaned = _clean_text(raw_text)
        if cleaned:
            pages.append(
                PageDocument(
                    text=cleaned,
                    source=filename,
                    page=page_number,
                    document_id=document_id,
                    metadata={"source": filename, "page": page_number, "document_id": document_id},
                )
            )

    if not pages:
        # Most likely a scanned/image-only PDF with no text layer.
        raise IngestionError(
            f"'{filename}' contains no extractable text. "
            "It may be a scanned/image-only PDF (OCR is not supported in V1)."
        )

    return pages


def _clean_text(text: str) -> str:
    """
    Light text normalization.

    PDF extraction often produces excessive whitespace, hyphenated
    line-breaks, and stray control characters. We do MINIMAL cleanup
    here (not aggressive rewriting) because aggressive cleanup can
    accidentally destroy meaning (e.g. merging words that should stay
    separate). Heavier normalization, if ever needed, belongs in
    chunking.py where we have more context.
    """
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)          # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse 3+ blank lines to 1
    return text.strip()


def load_pdfs(file_paths: list[str | Path]) -> tuple[list[PageDocument], list[str]]:
    """
    Load multiple PDFs, collecting successes and per-file errors
    separately so one bad file doesn't block the entire batch.

    Returns:
        (all_pages, errors) where `errors` is a list of human-readable
        error strings for files that failed, safe to show in the UI.
    """
    all_pages: list[PageDocument] = []
    errors: list[str] = []

    for path in file_paths:
        try:
            all_pages.extend(load_pdf(path))
        except IngestionError as exc:
            errors.append(str(exc))

    return all_pages, errors
