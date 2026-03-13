"""ReagentAI Equation Extractor Module.

Detects and extracts mathematical equations written in LaTeX notation from
academic paper text.  Supports the following notation styles:

* **Inline math** -- delimited by single dollar signs: ``$E = mc^2$``
* **Display math ($$)** -- delimited by double dollar signs: ``$$\\sum_i x_i$$``
* **Display math (brackets)** -- delimited by ``\\[`` ... ``\\]``
* **Display math (parens)** -- delimited by ``\\(`` ... ``\\)`` (sometimes
  used for inline in LaTeX source but treated as a match here)
* **Named environments** -- ``\\begin{equation}`` ... ``\\end{equation}``,
  ``\\begin{align}`` ... ``\\end{align}``, and similar.

Typical usage::

    from backend.parser.equation_extractor import EquationExtractor

    extractor = EquationExtractor()
    result = extractor.extract(full_text)
    for eq in result.equations:
        print(eq.equation_type, eq.latex)

Design decisions
----------------
* Extraction is purely regex-based so that it works without a full LaTeX
  compiler.  This means some edge cases (e.g. nested dollar signs inside
  ``\\text{}``) may produce false positives.  For the ReagentAI pipeline
  this is acceptable because downstream LLM agents can validate and filter.
* Each equation carries its character-level position in the source text so
  that callers can correlate equations with the section they belong to.
* A brief *context* window (surrounding text) is attached to every equation
  to aid downstream reasoning.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from backend.utils.logging import get_logger

logger = get_logger("backend.parser.equation_extractor")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EquationType(str, Enum):
    """Classification of the equation's delimiters.

    Members:
        INLINE: Single dollar-sign delimited (``$...$``).
        DISPLAY_DOLLAR: Double dollar-sign delimited (``$$...$$``).
        DISPLAY_BRACKET: Bracket delimited (``\\[...\\]``).
        DISPLAY_PAREN: Parenthesis delimited (``\\(...\\)``).
        ENVIRONMENT: LaTeX environment (``\\begin{equation}...\\end{equation}``).
    """

    INLINE = "inline"
    DISPLAY_DOLLAR = "display_dollar"
    DISPLAY_BRACKET = "display_bracket"
    DISPLAY_PAREN = "display_paren"
    ENVIRONMENT = "environment"


class ExtractedEquation(BaseModel):
    """A single equation extracted from the paper text.

    Attributes:
        latex: The raw LaTeX string (without surrounding delimiters).
        equation_type: What kind of delimiter was used.
        raw_match: The full matched string *including* delimiters.
        start_pos: Character offset of the match start in the source text.
        end_pos: Character offset of the match end in the source text.
        context_before: Up to *context_window* characters before the equation.
        context_after: Up to *context_window* characters after the equation.
        environment_name: Name of the LaTeX environment if applicable
            (e.g. ``"equation"``, ``"align"``).  ``None`` for non-environment
            matches.
    """

    latex: str = Field(..., description="LaTeX content without delimiters")
    equation_type: EquationType = Field(..., description="Delimiter classification")
    raw_match: str = Field(default="", description="Full match including delimiters")
    start_pos: int = Field(default=0, ge=0, description="Start offset in source text")
    end_pos: int = Field(default=0, ge=0, description="End offset in source text")
    context_before: str = Field(default="", description="Text before the equation")
    context_after: str = Field(default="", description="Text after the equation")
    environment_name: Optional[str] = Field(
        default=None,
        description="LaTeX environment name (equation, align, etc.)",
    )


class EquationExtractionResult(BaseModel):
    """Complete result of equation extraction from a paper.

    Attributes:
        equations: Ordered list of extracted equations.
        total_count: Total number of equations found.
        inline_count: Number of inline equations.
        display_count: Number of display equations (all display types).
        environment_count: Number of environment-based equations.
    """

    equations: list[ExtractedEquation] = Field(
        default_factory=list, description="All extracted equations"
    )
    total_count: int = Field(default=0, ge=0, description="Total equations found")
    inline_count: int = Field(default=0, ge=0, description="Inline equation count")
    display_count: int = Field(default=0, ge=0, description="Display equation count")
    environment_count: int = Field(default=0, ge=0, description="Environment equation count")


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# The order matters: more specific patterns (display, environment) must be
# tried *before* the generic inline pattern to avoid partial matches.

# LaTeX environments:  \begin{equation} ... \end{equation}  (and friends)
_ENV_NAMES = (
    r"equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|"
    r"flalign\*?|alignat\*?|split|array|cases|matrix|"
    r"bmatrix|pmatrix|vmatrix|Vmatrix"
)
_ENV_RE = re.compile(
    r"\\begin\{(?P<env>" + _ENV_NAMES + r")\}"
    r"(?P<body>.*?)"
    r"\\end\{(?P=env)\}",
    re.DOTALL,
)

# Display math: $$ ... $$
# We use a non-greedy match and require at least one non-whitespace char.
_DISPLAY_DOLLAR_RE = re.compile(
    r"\$\$(?P<body>[^\$]+?)\$\$",
    re.DOTALL,
)

# Display math: \[ ... \]
_DISPLAY_BRACKET_RE = re.compile(
    r"\\\[(?P<body>.*?)\\\]",
    re.DOTALL,
)

# Inline / display: \( ... \)
_DISPLAY_PAREN_RE = re.compile(
    r"\\\((?P<body>.*?)\\\)",
    re.DOTALL,
)

# Inline math: $ ... $
# Negative lookbehind for backslash (so escaped \$ is ignored).
# Negative lookbehind for another $ (so $$ is not matched).
# Require content between delimiters to have at least one non-space char.
_INLINE_RE = re.compile(
    r"(?<![\\$])"         # not preceded by \ or $
    r"\$"                 # opening $
    r"(?P<body>"
    r"[^\$\n]+"           # content: no $ or newline (inline stays on one line)
    r")"
    r"\$"                 # closing $
    r"(?!\$)",            # not followed by $ (would be $$)
)


# ---------------------------------------------------------------------------
# Extractor implementation
# ---------------------------------------------------------------------------


class EquationExtractor:
    """Extract LaTeX equations from academic paper text.

    The extractor runs multiple regex passes -- one per delimiter style --
    then merges and deduplicates the results by character position.

    Args:
        context_window: Number of characters of surrounding text to include
            with each equation for context.  Defaults to ``80``.

    Example::

        extractor = EquationExtractor(context_window=100)
        result = extractor.extract(paper_text)
        print(f"Found {result.total_count} equations")
    """

    def __init__(self, context_window: int = 80) -> None:
        self.context_window = max(0, context_window)

    def extract(self, text: str) -> EquationExtractionResult:
        """Extract all equations from *text*.

        Args:
            text: Full paper text (or a section of it).

        Returns:
            :class:`EquationExtractionResult` with all detected equations.
        """
        if not text or not text.strip():
            logger.warning("Empty text supplied to EquationExtractor")
            return EquationExtractionResult()

        logger.info("Starting equation extraction ({} chars)", len(text))

        raw_matches: list[tuple[int, int, str, str, EquationType, Optional[str]]] = []

        # Pass 1: LaTeX environments
        for m in _ENV_RE.finditer(text):
            raw_matches.append((
                m.start(),
                m.end(),
                m.group("body").strip(),
                m.group(0),
                EquationType.ENVIRONMENT,
                m.group("env"),
            ))

        # Pass 2: Display $$ ... $$
        for m in _DISPLAY_DOLLAR_RE.finditer(text):
            raw_matches.append((
                m.start(),
                m.end(),
                m.group("body").strip(),
                m.group(0),
                EquationType.DISPLAY_DOLLAR,
                None,
            ))

        # Pass 3: Display \[ ... \]
        for m in _DISPLAY_BRACKET_RE.finditer(text):
            raw_matches.append((
                m.start(),
                m.end(),
                m.group("body").strip(),
                m.group(0),
                EquationType.DISPLAY_BRACKET,
                None,
            ))

        # Pass 4: Display/Inline \( ... \)
        for m in _DISPLAY_PAREN_RE.finditer(text):
            raw_matches.append((
                m.start(),
                m.end(),
                m.group("body").strip(),
                m.group(0),
                EquationType.DISPLAY_PAREN,
                None,
            ))

        # Pass 5: Inline $ ... $
        for m in _INLINE_RE.finditer(text):
            raw_matches.append((
                m.start(),
                m.end(),
                m.group("body").strip(),
                m.group(0),
                EquationType.INLINE,
                None,
            ))

        # Deduplicate overlapping matches -- keep the longer/more-specific one
        merged = self._merge_overlapping(raw_matches)
        logger.debug("{} raw matches, {} after dedup", len(raw_matches), len(merged))

        # Build ExtractedEquation objects
        equations: list[ExtractedEquation] = []
        for start, end, latex, raw, eq_type, env_name in merged:
            if not latex:
                continue
            # Skip trivially short matches that are likely noise
            if len(latex) < 2 and eq_type == EquationType.INLINE:
                continue
            # Skip matches that look like currency ($5, $100)
            if eq_type == EquationType.INLINE and self._looks_like_currency(latex):
                continue

            ctx_before = text[max(0, start - self.context_window): start].strip()
            ctx_after = text[end: end + self.context_window].strip()

            equations.append(
                ExtractedEquation(
                    latex=latex,
                    equation_type=eq_type,
                    raw_match=raw,
                    start_pos=start,
                    end_pos=end,
                    context_before=ctx_before,
                    context_after=ctx_after,
                    environment_name=env_name,
                )
            )

        # Compute counts
        inline_count = sum(1 for e in equations if e.equation_type == EquationType.INLINE)
        env_count = sum(1 for e in equations if e.equation_type == EquationType.ENVIRONMENT)
        display_count = len(equations) - inline_count - env_count

        result = EquationExtractionResult(
            equations=equations,
            total_count=len(equations),
            inline_count=inline_count,
            display_count=display_count,
            environment_count=env_count,
        )

        logger.info(
            "Equation extraction complete: {} total ({} inline, {} display, {} environment)",
            result.total_count,
            result.inline_count,
            result.display_count,
            result.environment_count,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_overlapping(
        matches: list[tuple[int, int, str, str, EquationType, Optional[str]]],
    ) -> list[tuple[int, int, str, str, EquationType, Optional[str]]]:
        """Remove overlapping matches, keeping the longer one.

        When two matches overlap in character range, the one spanning more
        characters wins.  In case of a tie, the one detected first (earlier
        pass = higher specificity) wins.

        Args:
            matches: Raw match tuples ``(start, end, latex, raw, type, env)``.

        Returns:
            Filtered and sorted list of non-overlapping matches.
        """
        if not matches:
            return []

        # Sort by start position, then by length descending (prefer longer)
        sorted_matches = sorted(matches, key=lambda m: (m[0], -(m[1] - m[0])))

        result: list[tuple[int, int, str, str, EquationType, Optional[str]]] = []
        last_end = -1

        for item in sorted_matches:
            start, end = item[0], item[1]
            if start >= last_end:
                result.append(item)
                last_end = end

        return result

    @staticmethod
    def _looks_like_currency(latex: str) -> bool:
        """Heuristic check: does the content look like a currency amount?

        Catches things like ``$100`` being parsed as inline math ``100``.

        Args:
            latex: The extracted LaTeX content (without delimiters).

        Returns:
            ``True`` if the content looks like a plain number / currency.
        """
        stripped = latex.strip()
        # Pure integer or decimal number
        if re.fullmatch(r"\d[\d,]*\.?\d*", stripped):
            return True
        # Number with trailing unit-like suffix (e.g. "100M", "3.5B")
        if re.fullmatch(r"\d[\d,]*\.?\d*\s*[KMBTkmbt]", stripped):
            return True
        return False
