"""ReagentAI PDF Parser Module.

Extracts full text, per-page content, and metadata from academic PDF papers
using PyMuPDF (fitz). Produces a structured ``PaperDocument`` that downstream
modules (section extraction, equation extraction) consume.

Typical usage::

    from backend.parser.pdf_parser import PDFParser

    parser = PDFParser()
    document = parser.parse("path/to/paper.pdf")
    print(document.full_text[:500])

Design decisions
----------------
* **PyMuPDF** is the primary extraction engine because it is fast, handles
  most PDF encodings well, and exposes both text and metadata in a single pass.
* Every public method returns Pydantic models so that the rest of the pipeline
  can rely on validated, JSON-serialisable data.
* Logging is performed through ``loguru`` via the project's central logging
  utility so that all parser activity appears in the unified log stream.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF

from pydantic import BaseModel, Field

from backend.utils.logging import get_logger

logger = get_logger("backend.parser.pdf_parser")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PageContent(BaseModel):
    """Represents the extracted content of a single PDF page.

    Attributes:
        page_number: 1-based page index.
        text: Raw text extracted from the page.
        char_count: Number of characters on this page.
        word_count: Approximate word count on this page.
    """

    page_number: int = Field(..., ge=1, description="1-based page number")
    text: str = Field(default="", description="Raw extracted text for this page")
    char_count: int = Field(default=0, ge=0, description="Character count")
    word_count: int = Field(default=0, ge=0, description="Approximate word count")


class PaperMetadata(BaseModel):
    """Bibliographic and technical metadata extracted from the PDF.

    Attributes:
        title: Paper title (from PDF metadata or heuristic extraction).
        authors: List of author names when available.
        page_count: Total number of pages.
        creation_date: PDF creation timestamp if present.
        modification_date: PDF modification timestamp if present.
        producer: Software that produced the PDF.
        creator: Software that created the source document.
        subject: PDF subject field.
        keywords: PDF keywords field.
        format: PDF version string reported by the file.
    """

    title: str = Field(default="", description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Author names")
    page_count: int = Field(default=0, ge=0, description="Total pages")
    creation_date: Optional[str] = Field(default=None, description="PDF creation date")
    modification_date: Optional[str] = Field(default=None, description="PDF modification date")
    producer: str = Field(default="", description="PDF producer software")
    creator: str = Field(default="", description="PDF creator software")
    subject: str = Field(default="", description="PDF subject")
    keywords: str = Field(default="", description="PDF keywords")
    format: str = Field(default="", description="PDF format version")


class PaperDocument(BaseModel):
    """Complete structured representation of a parsed PDF paper.

    This is the primary output of :class:`PDFParser` and serves as the
    canonical input for all downstream extraction modules.

    Attributes:
        metadata: Bibliographic/technical metadata.
        pages: Ordered list of per-page content.
        full_text: Concatenation of all page texts.
        file_path: Absolute path to the source PDF.
        file_size_bytes: Size of the source PDF in bytes.
        parse_timestamp: ISO-8601 timestamp of when parsing occurred.
        total_char_count: Total character count across all pages.
        total_word_count: Total word count across all pages.
    """

    metadata: PaperMetadata = Field(default_factory=PaperMetadata)
    pages: list[PageContent] = Field(default_factory=list)
    full_text: str = Field(default="", description="Full concatenated text")
    file_path: str = Field(default="", description="Absolute source path")
    file_size_bytes: int = Field(default=0, ge=0, description="File size in bytes")
    parse_timestamp: str = Field(default="", description="ISO-8601 parse time")
    total_char_count: int = Field(default=0, ge=0)
    total_word_count: int = Field(default=0, ge=0)

    def to_json(self) -> str:
        """Serialise the document to a JSON string.

        Returns:
            Pretty-printed JSON representation of the document.
        """
        return self.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Parser implementation
# ---------------------------------------------------------------------------


class PDFParser:
    """Extracts text and metadata from academic PDFs using PyMuPDF.

    The parser is stateless -- each call to :meth:`parse` processes a single
    file independently, making it safe to reuse across requests.

    Example::

        parser = PDFParser()
        doc = parser.parse("/data/papers/attention.pdf")
    """

    # Characters used to join pages when building ``full_text``.
    PAGE_SEPARATOR: str = "\n\n"

    def parse(self, pdf_path: str | Path) -> PaperDocument:
        """Parse a PDF file and return a structured :class:`PaperDocument`.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            A fully populated :class:`PaperDocument`.

        Raises:
            FileNotFoundError: If *pdf_path* does not exist.
            ValueError: If the file is not a valid PDF or cannot be opened.
        """
        pdf_path = Path(pdf_path).resolve()
        logger.info("Starting PDF parse: {}", pdf_path)

        if not pdf_path.exists():
            logger.error("PDF file not found: {}", pdf_path)
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            logger.warning("File does not have .pdf extension: {}", pdf_path)

        start_time = time.monotonic()

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.error("Failed to open PDF {}: {}", pdf_path, exc)
            raise ValueError(f"Cannot open PDF file: {pdf_path}") from exc

        try:
            metadata = self._extract_metadata(doc)
            pages = self._extract_pages(doc)
        finally:
            doc.close()

        full_text = self.PAGE_SEPARATOR.join(page.text for page in pages)
        total_chars = sum(page.char_count for page in pages)
        total_words = sum(page.word_count for page in pages)
        file_size = os.path.getsize(pdf_path)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Parsed {} pages ({} chars, {} words) from {} in {:.2f}s",
            len(pages),
            total_chars,
            total_words,
            pdf_path.name,
            elapsed,
        )

        # If metadata title is empty, try heuristic title extraction
        if not metadata.title:
            metadata.title = self._heuristic_title(pages)

        return PaperDocument(
            metadata=metadata,
            pages=pages,
            full_text=full_text,
            file_path=str(pdf_path),
            file_size_bytes=file_size,
            parse_timestamp=datetime.utcnow().isoformat() + "Z",
            total_char_count=total_chars,
            total_word_count=total_words,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_metadata(self, doc: fitz.Document) -> PaperMetadata:
        """Extract metadata from an open PyMuPDF document.

        Args:
            doc: An open ``fitz.Document`` instance.

        Returns:
            Populated :class:`PaperMetadata`.
        """
        raw: dict[str, Any] = doc.metadata or {}
        logger.debug("Raw PDF metadata keys: {}", list(raw.keys()))

        title = (raw.get("title") or "").strip()
        author_str = (raw.get("author") or "").strip()

        # Split authors on common delimiters: comma, semicolon, " and "
        authors: list[str] = []
        if author_str:
            import re

            parts = re.split(r"[;,]|\band\b", author_str)
            authors = [a.strip() for a in parts if a.strip()]

        creation_date = self._parse_pdf_date(raw.get("creationDate"))
        mod_date = self._parse_pdf_date(raw.get("modDate"))

        return PaperMetadata(
            title=title,
            authors=authors,
            page_count=doc.page_count,
            creation_date=creation_date,
            modification_date=mod_date,
            producer=(raw.get("producer") or "").strip(),
            creator=(raw.get("creator") or "").strip(),
            subject=(raw.get("subject") or "").strip(),
            keywords=(raw.get("keywords") or "").strip(),
            format=(raw.get("format") or "").strip(),
        )

    def _extract_pages(self, doc: fitz.Document) -> list[PageContent]:
        """Extract text from every page in the document.

        Args:
            doc: An open ``fitz.Document`` instance.

        Returns:
            Ordered list of :class:`PageContent` instances, one per page.
        """
        pages: list[PageContent] = []

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            text = page.get_text("text") or ""
            text = self._clean_page_text(text)
            char_count = len(text)
            word_count = len(text.split()) if text else 0

            pages.append(
                PageContent(
                    page_number=page_idx + 1,
                    text=text,
                    char_count=char_count,
                    word_count=word_count,
                )
            )
            logger.debug(
                "Page {}: {} chars, {} words", page_idx + 1, char_count, word_count
            )

        return pages

    @staticmethod
    def _clean_page_text(text: str) -> str:
        """Apply light normalisation to raw page text.

        * Collapses runs of three or more newlines into two.
        * Strips trailing whitespace from every line.

        Args:
            text: Raw extracted page text.

        Returns:
            Cleaned text.
        """
        import re

        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)

    @staticmethod
    def _parse_pdf_date(raw_date: Optional[str]) -> Optional[str]:
        """Convert a PDF date string (``D:YYYYMMDDHHmmSS...``) to ISO-8601.

        Args:
            raw_date: Raw date string from PDF metadata.

        Returns:
            ISO-8601 formatted date string, or ``None`` if parsing fails.
        """
        if not raw_date:
            return None

        import re

        # Strip the ``D:`` prefix that PDF dates conventionally carry.
        cleaned = raw_date.strip()
        if cleaned.startswith("D:"):
            cleaned = cleaned[2:]

        # Remove timezone offset characters for basic parsing.
        cleaned = re.sub(r"[Z+'\\-].*$", "", cleaned)

        try:
            if len(cleaned) >= 14:
                dt = datetime.strptime(cleaned[:14], "%Y%m%d%H%M%S")
            elif len(cleaned) >= 8:
                dt = datetime.strptime(cleaned[:8], "%Y%m%d")
            else:
                return raw_date  # Return original if format is unknown
            return dt.isoformat() + "Z"
        except ValueError:
            return raw_date

    @staticmethod
    def _heuristic_title(pages: list[PageContent]) -> str:
        """Attempt to guess the paper title from the first page text.

        Heuristic: the title is usually the first non-empty line on page 1
        that is longer than 10 characters and shorter than 300 characters.

        Args:
            pages: Extracted pages list.

        Returns:
            Best-guess title string, or empty string.
        """
        if not pages:
            return ""

        first_page = pages[0].text
        for line in first_page.split("\n"):
            stripped = line.strip()
            if 10 < len(stripped) < 300 and not stripped.startswith("http"):
                logger.debug("Heuristic title: '{}'", stripped)
                return stripped

        return ""
