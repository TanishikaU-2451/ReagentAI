"""ReagentAI Section Extractor Module.

Detects and extracts named sections from the full text of an academic paper.
Uses a combination of regex pattern matching and heuristic rules to identify
standard sections found in ML / AI conference papers:

* Title
* Abstract
* Introduction
* Related Work
* Methodology / Methods / Approach
* Dataset / Data
* Experiments / Results / Evaluation
* Discussion
* Conclusion
* References / Bibliography
* Appendix

Typical usage::

    from backend.parser.section_extractor import SectionExtractor

    extractor = SectionExtractor()
    sections = extractor.extract(full_text)
    print(sections.sections.keys())

Design decisions
----------------
* Section headers in academic papers appear in many styles (numbered, bold,
  ALL-CAPS, Title Case).  The detector normalises lines before matching to
  cover these variants.
* Because heuristics can never be perfect, the extractor is deliberately
  lenient -- it prefers returning a slightly too-large section over missing
  content.
* The ``abstract`` is also detected by the common "Abstract" keyword that
  appears at the start of most papers even when it is not a numbered section.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from backend.utils.logging import get_logger

logger = get_logger("backend.parser.section_extractor")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SectionContent(BaseModel):
    """A single extracted section of the paper.

    Attributes:
        name: Canonical section name (e.g. ``"abstract"``, ``"methodology"``).
        raw_header: The original header line as it appeared in the text.
        text: Body text of the section (header line excluded).
        start_pos: Character offset where the section starts in the full text.
        end_pos: Character offset where the section ends in the full text.
        word_count: Approximate word count of the section body.
    """

    name: str = Field(..., description="Canonical section name")
    raw_header: str = Field(default="", description="Original header text")
    text: str = Field(default="", description="Section body text")
    start_pos: int = Field(default=0, ge=0, description="Start char offset")
    end_pos: int = Field(default=0, ge=0, description="End char offset")
    word_count: int = Field(default=0, ge=0, description="Approximate word count")


class PaperSections(BaseModel):
    """Collection of all sections extracted from a paper.

    Attributes:
        sections: Mapping of canonical section name to :class:`SectionContent`.
        section_order: Ordered list of canonical section names as they appear.
        unclassified_text: Text that was not assigned to any section.
    """

    sections: dict[str, SectionContent] = Field(
        default_factory=dict,
        description="Section name -> content mapping",
    )
    section_order: list[str] = Field(
        default_factory=list,
        description="Sections in order of appearance",
    )
    unclassified_text: str = Field(
        default="",
        description="Text not belonging to any detected section",
    )

    def get_section(self, name: str) -> Optional[SectionContent]:
        """Return a section by its canonical name, or ``None``.

        Args:
            name: Canonical section name (case-insensitive).

        Returns:
            :class:`SectionContent` or ``None``.
        """
        return self.sections.get(name.lower())

    def to_dict(self) -> dict[str, str]:
        """Flatten sections to a simple ``{name: text}`` mapping.

        Returns:
            Dict mapping canonical section names to their text content.
        """
        return {name: sec.text for name, sec in self.sections.items()}


# ---------------------------------------------------------------------------
# Canonical section definitions
# ---------------------------------------------------------------------------

# Each entry: (canonical_name, list_of_regex_patterns)
# Patterns are matched against a *normalised* header line (lowercased,
# leading numbers/punctuation stripped).
_SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "abstract",
        [
            r"^abstract$",
            r"^summary$",
        ],
    ),
    (
        "introduction",
        [
            r"^introduction$",
            r"^background$",
            r"^overview$",
        ],
    ),
    (
        "related_work",
        [
            r"^related\s+work$",
            r"^prior\s+work$",
            r"^literature\s+review$",
            r"^previous\s+work$",
        ],
    ),
    (
        "methodology",
        [
            r"^method(?:ology)?$",
            r"^methods?$",
            r"^approach$",
            r"^proposed\s+(?:method|approach|framework|model|system)$",
            r"^(?:our|the)\s+(?:method|approach|framework|model|system)$",
            r"^model(?:\s+architecture)?$",
            r"^framework$",
            r"^technical\s+approach$",
            r"^algorithm$",
        ],
    ),
    (
        "dataset",
        [
            r"^data(?:set)?s?$",
            r"^data\s+(?:collection|description|preparation|preprocessing)$",
            r"^benchmark(?:s)?$",
            r"^corpus$",
            r"^experimental\s+setup$",
            r"^setup$",
        ],
    ),
    (
        "experiments",
        [
            r"^experiments?$",
            r"^results?$",
            r"^experiments?\s+and\s+results?$",
            r"^results?\s+and\s+(?:discussion|analysis)$",
            r"^evaluation$",
            r"^empirical\s+(?:evaluation|results|analysis)$",
            r"^analysis$",
            r"^experimental\s+results?$",
            r"^ablation\s+stud(?:y|ies)$",
        ],
    ),
    (
        "discussion",
        [
            r"^discussion$",
            r"^discussion\s+and\s+(?:future\s+work|limitations)$",
            r"^limitations$",
            r"^limitations?\s+and\s+future\s+work$",
        ],
    ),
    (
        "conclusion",
        [
            r"^conclusions?$",
            r"^concluding\s+remarks$",
            r"^conclusion\s+and\s+future\s+work$",
            r"^summary\s+and\s+conclusions?$",
            r"^final\s+remarks$",
        ],
    ),
    (
        "references",
        [
            r"^references$",
            r"^bibliography$",
        ],
    ),
    (
        "appendix",
        [
            r"^appendi(?:x|ces)$",
            r"^supplementary\s+material$",
            r"^supplementary$",
        ],
    ),
]

# Compiled patterns for each canonical section name.
_COMPILED_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    (name, [re.compile(p, re.IGNORECASE) for p in patterns])
    for name, patterns in _SECTION_PATTERNS
]

# ---------------------------------------------------------------------------
# Regex for detecting section-like header lines in academic papers.
# ---------------------------------------------------------------------------

# A line that looks like a section header:
#   - Optional leading numbering  (1. / 1 / I. / A. / 1.2 / etc.)
#   - Followed by mainly alphabetic words
#   - Total length typically under 120 characters
_HEADER_LINE_RE = re.compile(
    r"^"
    r"(?:"
    r"(?:\d{1,2}(?:\.\d{1,2})*\.?[ \t]+)"  # e.g. "1. " or "3.2 "
    r"|(?:[IVXLC]+\.?[ \t]+)"               # e.g. "IV " or "III. "
    r"|(?:[A-Z]\.?[ \t]+)"                   # e.g. "A. " or "B "
    r")?"
    r"(?P<heading>[A-Za-z][\w :&,/\\-]{2,118})"
    r"$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Extractor implementation
# ---------------------------------------------------------------------------


class SectionExtractor:
    """Heuristic section detector for academic paper text.

    The extractor performs two passes:

    1. **Header detection** -- Every line in the text is tested against a set
       of regex patterns to determine whether it looks like a section header.
    2. **Section assignment** -- Consecutive text between detected headers is
       assigned to the preceding header's canonical section name.

    The ``abstract`` section receives special treatment because it is often
    not preceded by a numbered header.

    Example::

        extractor = SectionExtractor()
        sections = extractor.extract(paper_full_text)
    """

    def extract(self, text: str) -> PaperSections:
        """Extract sections from the full text of a paper.

        Args:
            text: Complete paper text (e.g. ``PaperDocument.full_text``).

        Returns:
            :class:`PaperSections` containing all detected sections.
        """
        if not text or not text.strip():
            logger.warning("Empty text supplied to SectionExtractor")
            return PaperSections()

        logger.info("Starting section extraction ({} chars)", len(text))

        # Step 1: detect header positions
        headers = self._detect_headers(text)
        logger.debug("Detected {} candidate headers", len(headers))

        # Step 2: try to extract abstract specially (it is often unlabeled
        # or uses a different visual cue)
        abstract_section = self._extract_abstract(text, headers)

        # Step 3: build sections from detected headers
        sections: dict[str, SectionContent] = {}
        section_order: list[str] = []

        if abstract_section:
            sections["abstract"] = abstract_section
            section_order.append("abstract")

        for idx, (canonical, raw_header, start) in enumerate(headers):
            # Determine the end position: start of next header or end of text
            if idx + 1 < len(headers):
                end = headers[idx + 1][2]
            else:
                end = len(text)

            # Body starts right after the header line
            header_end = start + len(raw_header)
            body = text[header_end:end].strip()
            word_count = len(body.split()) if body else 0

            # Skip abstract if we already extracted it specially
            if canonical == "abstract" and "abstract" in sections:
                # Update position info if the labeled version is better
                if word_count > sections["abstract"].word_count:
                    sections["abstract"] = SectionContent(
                        name="abstract",
                        raw_header=raw_header,
                        text=body,
                        start_pos=start,
                        end_pos=end,
                        word_count=word_count,
                    )
                continue

            # If we already have this section, keep the first occurrence
            if canonical in sections:
                logger.debug(
                    "Duplicate section '{}' at pos {}; keeping first", canonical, start
                )
                continue

            sections[canonical] = SectionContent(
                name=canonical,
                raw_header=raw_header,
                text=body,
                start_pos=start,
                end_pos=end,
                word_count=word_count,
            )
            if canonical not in section_order:
                section_order.append(canonical)

        # Compute unclassified text (anything before the first header)
        unclassified = ""
        if headers:
            first_start = headers[0][2]
            preamble = text[:first_start].strip()
            # Remove abstract from preamble if we extracted it
            if abstract_section and preamble:
                preamble = preamble.replace(abstract_section.text, "").strip()
            unclassified = preamble
        else:
            unclassified = text.strip()

        result = PaperSections(
            sections=sections,
            section_order=section_order,
            unclassified_text=unclassified,
        )

        logger.info(
            "Section extraction complete: {} sections found ({})",
            len(sections),
            ", ".join(section_order),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_headers(
        self, text: str
    ) -> list[tuple[str, str, int]]:
        """Scan the text for lines that look like section headers.

        Args:
            text: Full paper text.

        Returns:
            List of ``(canonical_name, raw_header_line, char_offset)`` tuples
            sorted by char_offset.
        """
        results: list[tuple[str, str, int]] = []

        for match in _HEADER_LINE_RE.finditer(text):
            raw_line = match.group(0).strip()
            heading_text = match.group("heading").strip()
            normalised = self._normalise_heading(heading_text)

            canonical = self._classify_heading(normalised)
            if canonical is not None:
                results.append((canonical, raw_line, match.start()))
                logger.debug(
                    "Header '{}' -> '{}' at pos {}", raw_line, canonical, match.start()
                )

        # Also check for ALL-CAPS headings not caught by the main regex
        allcaps_results = self._detect_allcaps_headers(text)
        results.extend(allcaps_results)

        # Deduplicate and sort by position
        seen_positions: set[int] = set()
        unique: list[tuple[str, str, int]] = []
        for item in sorted(results, key=lambda x: x[2]):
            if item[2] not in seen_positions:
                seen_positions.add(item[2])
                unique.append(item)

        return unique

    def _detect_allcaps_headers(
        self, text: str
    ) -> list[tuple[str, str, int]]:
        """Detect ALL-CAPS section headers.

        Many conference papers (e.g. IEEE format) use fully capitalised
        section headings like ``INTRODUCTION`` or ``RELATED WORK``.

        Args:
            text: Full paper text.

        Returns:
            List of ``(canonical_name, raw_line, char_offset)`` tuples.
        """
        results: list[tuple[str, str, int]] = []
        allcaps_re = re.compile(
            r"^(?:\d{1,2}\.?[ \t]+)?(?P<heading>[A-Z][A-Z &,/-]{2,60})$",
            re.MULTILINE,
        )
        for match in allcaps_re.finditer(text):
            heading = match.group("heading").strip()
            # Validate it is truly all caps (not just an acronym line)
            words = heading.split()
            if len(words) < 1 or len(words) > 8:
                continue
            # At least one word must be >= 4 chars (filters out short acronyms)
            if not any(len(w) >= 4 for w in words):
                continue

            normalised = self._normalise_heading(heading)
            canonical = self._classify_heading(normalised)
            if canonical is not None:
                results.append((canonical, match.group(0).strip(), match.start()))

        return results

    def _extract_abstract(
        self,
        text: str,
        headers: list[tuple[str, str, int]],
    ) -> Optional[SectionContent]:
        """Try to extract the abstract using keyword-based heuristics.

        Many papers print the abstract before any numbered section, preceded
        by the word "Abstract" on its own line or as a run-in label.

        Args:
            text: Full paper text.
            headers: Already-detected headers (to determine boundaries).

        Returns:
            :class:`SectionContent` for the abstract, or ``None``.
        """
        # Pattern 1: "Abstract" or "ABSTRACT" on its own line
        abstract_re = re.compile(
            r"(?:^|\n)\s*(?:abstract|ABSTRACT)\s*[:\.\-]?\s*\n",
            re.IGNORECASE,
        )
        match = abstract_re.search(text)
        if match:
            start = match.end()
            # End at the next detected header or after ~5000 chars max
            end = self._find_abstract_end(text, start, headers)
            body = text[start:end].strip()
            if body and len(body) > 20:
                return SectionContent(
                    name="abstract",
                    raw_header=match.group(0).strip(),
                    text=body,
                    start_pos=match.start(),
                    end_pos=end,
                    word_count=len(body.split()),
                )

        # Pattern 2: "Abstract." or "Abstract:" as run-in prefix
        runin_re = re.compile(
            r"(?:^|\n)\s*(?:abstract|ABSTRACT)\s*[:\.\-]\s*(?P<body>.+)",
            re.IGNORECASE,
        )
        match = runin_re.search(text)
        if match:
            start = match.start("body")
            end = self._find_abstract_end(text, start, headers)
            body = text[start:end].strip()
            if body and len(body) > 20:
                return SectionContent(
                    name="abstract",
                    raw_header="Abstract",
                    text=body,
                    start_pos=match.start(),
                    end_pos=end,
                    word_count=len(body.split()),
                )

        return None

    @staticmethod
    def _find_abstract_end(
        text: str,
        start: int,
        headers: list[tuple[str, str, int]],
    ) -> int:
        """Determine where the abstract ends.

        Args:
            text: Full paper text.
            start: Character offset where abstract body begins.
            headers: Detected headers (sorted by position).

        Returns:
            Character offset for the end of the abstract.
        """
        # End at the first header that comes after start
        for _, _, hdr_pos in headers:
            if hdr_pos > start:
                return hdr_pos

        # Fallback: limit to ~5000 chars after start
        return min(start + 5000, len(text))

    @staticmethod
    def _normalise_heading(heading: str) -> str:
        """Normalise a heading string for pattern matching.

        * Lowercase
        * Strip leading/trailing whitespace and punctuation
        * Collapse internal whitespace

        Args:
            heading: Raw heading text.

        Returns:
            Normalised heading string.
        """
        h = heading.lower().strip()
        # Remove trailing punctuation (colon, period, dash)
        h = re.sub(r"[\s:.\-]+$", "", h)
        # Remove leading numbering artefacts
        h = re.sub(r"^[\d.\s]+", "", h)
        h = re.sub(r"^[ivxlc]+\.?\s+", "", h, flags=re.IGNORECASE)
        # Collapse whitespace
        h = re.sub(r"\s+", " ", h).strip()
        return h

    @staticmethod
    def _classify_heading(normalised: str) -> Optional[str]:
        """Match a normalised heading against known section patterns.

        Args:
            normalised: Normalised heading text.

        Returns:
            Canonical section name, or ``None`` if no match.
        """
        for canonical, patterns in _COMPILED_PATTERNS:
            for pat in patterns:
                if pat.match(normalised):
                    return canonical
        return None
