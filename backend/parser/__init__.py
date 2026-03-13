"""ReagentAI Paper Parsing Package -- Phase 1.

This package provides end-to-end parsing of academic PDF papers into a
structured JSON representation.  It is composed of three modules:

* :mod:`backend.parser.pdf_parser` -- PDF text and metadata extraction
  (PyMuPDF).
* :mod:`backend.parser.section_extractor` -- Heuristic section detection
  (title, abstract, methodology, etc.).
* :mod:`backend.parser.equation_extractor` -- LaTeX equation extraction
  (inline and display math).

Quick start
-----------

The simplest entry point is the top-level :func:`parse_paper` function::

    from backend.parser import parse_paper

    result = parse_paper("path/to/paper.pdf")
    print(result["metadata"]["title"])
    print(result["sections"].keys())
    print(len(result["equations"]))

All public classes and models are re-exported from this ``__init__`` for
convenience::

    from backend.parser import PDFParser, SectionExtractor, EquationExtractor
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.parser.equation_extractor import (
    EquationExtractionResult,
    EquationExtractor,
    EquationType,
    ExtractedEquation,
)
from backend.parser.pdf_parser import (
    PageContent,
    PaperDocument,
    PaperMetadata,
    PDFParser,
)
from backend.parser.section_extractor import (
    PaperSections,
    SectionContent,
    SectionExtractor,
)
from backend.utils.logging import get_logger

logger = get_logger("backend.parser")

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    # High-level API
    "parse_paper",
    "ParsedPaper",
    # PDF parsing
    "PDFParser",
    "PaperDocument",
    "PaperMetadata",
    "PageContent",
    # Section extraction
    "SectionExtractor",
    "PaperSections",
    "SectionContent",
    # Equation extraction
    "EquationExtractor",
    "EquationExtractionResult",
    "ExtractedEquation",
    "EquationType",
]


# ---------------------------------------------------------------------------
# Unified output model
# ---------------------------------------------------------------------------


class ParsedPaper(BaseModel):
    """Unified structured output from the full parsing pipeline.

    Combines the outputs of PDF parsing, section extraction, and equation
    extraction into a single JSON-serialisable object.

    Attributes:
        document: The raw parsed PDF document.
        sections: Detected paper sections.
        equations: Extracted mathematical equations.
        pipeline_elapsed_seconds: Wall-clock time for the entire pipeline.
    """

    document: PaperDocument = Field(
        default_factory=PaperDocument,
        description="Raw PDF extraction result",
    )
    sections: PaperSections = Field(
        default_factory=PaperSections,
        description="Detected paper sections",
    )
    equations: EquationExtractionResult = Field(
        default_factory=EquationExtractionResult,
        description="Extracted LaTeX equations",
    )
    pipeline_elapsed_seconds: float = Field(
        default=0.0,
        description="Total pipeline wall-clock time in seconds",
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full parsed paper to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary.

        Returns:
            Nested dictionary representation.
        """
        return self.model_dump()

    def summary(self) -> dict[str, Any]:
        """Return a concise summary of the parsing results.

        Useful for logging and quick inspection without dumping the full
        text content.

        Returns:
            Dictionary with key statistics.
        """
        return {
            "file_path": self.document.file_path,
            "title": self.document.metadata.title,
            "authors": self.document.metadata.authors,
            "page_count": self.document.metadata.page_count,
            "total_chars": self.document.total_char_count,
            "total_words": self.document.total_word_count,
            "sections_found": self.sections.section_order,
            "equation_count": self.equations.total_count,
            "inline_equations": self.equations.inline_count,
            "display_equations": self.equations.display_count,
            "environment_equations": self.equations.environment_count,
            "pipeline_seconds": round(self.pipeline_elapsed_seconds, 3),
        }


# ---------------------------------------------------------------------------
# High-level pipeline function
# ---------------------------------------------------------------------------


def parse_paper(
    pdf_path: str | Path,
    *,
    extract_sections: bool = True,
    extract_equations: bool = True,
    equation_context_window: int = 80,
) -> ParsedPaper:
    """Run the full Phase 1 parsing pipeline on a PDF paper.

    This is the recommended entry point for consumers of the parsing
    package.  It orchestrates PDF extraction, section detection, and
    equation extraction, returning a single :class:`ParsedPaper` object.

    Args:
        pdf_path: Path to the PDF file.
        extract_sections: Whether to run section extraction.  Defaults to
            ``True``.
        extract_equations: Whether to run equation extraction.  Defaults to
            ``True``.
        equation_context_window: Characters of context to capture around each
            equation.

    Returns:
        A :class:`ParsedPaper` containing the full structured output.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        ValueError: If the PDF cannot be opened or parsed.

    Example::

        result = parse_paper("attention_is_all_you_need.pdf")
        print(result.summary())
        with open("parsed.json", "w") as f:
            f.write(result.to_json())
    """
    pdf_path = Path(pdf_path).resolve()
    logger.info("=== Phase 1 Paper Parsing Pipeline ===")
    logger.info("Input: {}", pdf_path)
    pipeline_start = time.monotonic()

    # Step 1: PDF extraction
    logger.info("[1/3] Extracting text and metadata from PDF")
    pdf_parser = PDFParser()
    document = pdf_parser.parse(pdf_path)

    # Step 2: Section extraction
    sections = PaperSections()
    if extract_sections:
        logger.info("[2/3] Detecting paper sections")
        section_extractor = SectionExtractor()
        sections = section_extractor.extract(document.full_text)
    else:
        logger.info("[2/3] Section extraction skipped (disabled)")

    # Step 3: Equation extraction
    equations = EquationExtractionResult()
    if extract_equations:
        logger.info("[3/3] Extracting mathematical equations")
        equation_extractor = EquationExtractor(context_window=equation_context_window)
        equations = equation_extractor.extract(document.full_text)
    else:
        logger.info("[3/3] Equation extraction skipped (disabled)")

    elapsed = time.monotonic() - pipeline_start

    result = ParsedPaper(
        document=document,
        sections=sections,
        equations=equations,
        pipeline_elapsed_seconds=round(elapsed, 4),
    )

    logger.info(
        "Pipeline complete in {:.2f}s -- {} pages, {} sections, {} equations",
        elapsed,
        document.metadata.page_count,
        len(sections.sections),
        equations.total_count,
    )
    logger.debug("Summary: {}", result.summary())

    return result
