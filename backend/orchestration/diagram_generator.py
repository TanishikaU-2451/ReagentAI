"""ReagentAI Architecture Diagram Generation -- Phase 10.

Orchestrates the generation of Mermaid.js diagrams that visualise the
ML model architecture, training pipeline, and data flow of a generated
project.  Delegates LLM-based diagram generation to the
:class:`~backend.agents.diagram_agent.DiagramAgent` and provides
post-processing utilities for wrapping the raw Mermaid code in
frontend-friendly structures.

Three diagram types are produced:

* **model_architecture** -- Block diagram of the neural network layers,
  connections, and tensor shapes.
* **training_pipeline** -- Flowchart showing the end-to-end training loop
  (data loading, forward pass, loss, backward pass, optimiser step).
* **data_flow** -- Diagram depicting how data moves from raw input
  through preprocessing, batching, and into the model.

Usage::

    from backend.orchestration.diagram_generator import DiagramGenerator

    generator = DiagramGenerator()
    result = await generator.generate(
        research_summary={...},
        source_files={"src/model.py": "...", ...},
    )
    for dtype, diagram in result.items():
        print(dtype, diagram["mermaid"][:80])
"""

from __future__ import annotations

import re
import textwrap
import time
from typing import Any, Dict, List, Optional

from backend.agents import DiagramAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger

logger = get_logger("orchestration.diagram_generator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The three diagram types we generate
DIAGRAM_TYPES: List[str] = [
    "model_architecture",
    "training_pipeline",
    "data_flow",
]

# Default Mermaid theme used when wrapping for frontend rendering
_DEFAULT_MERMAID_THEME = "default"

# Mermaid HTML wrapper template for frontend rendering
_MERMAID_HTML_TEMPLATE = textwrap.dedent("""\
    <div class="mermaid-diagram" data-diagram-type="{diagram_type}">
      <h3>{title}</h3>
      <pre class="mermaid">
    {mermaid_code}
      </pre>
    </div>
""")

# Mermaid Markdown wrapper template
_MERMAID_MD_TEMPLATE = textwrap.dedent("""\
    ### {title}

    ```mermaid
    {mermaid_code}
    ```
""")


# ---------------------------------------------------------------------------
# Diagram result container
# ---------------------------------------------------------------------------

class DiagramResult:
    """Container for a single generated diagram.

    Attributes:
        diagram_type: One of ``model_architecture``, ``training_pipeline``,
                      or ``data_flow``.
        title:        Human-readable title for the diagram.
        mermaid_code: Raw Mermaid.js source code.
        generation_time: Wall-clock seconds taken to generate this diagram.
    """

    __slots__ = ("diagram_type", "title", "mermaid_code", "generation_time")

    def __init__(
        self,
        diagram_type: str,
        title: str,
        mermaid_code: str,
        generation_time: float = 0.0,
    ) -> None:
        self.diagram_type = diagram_type
        self.title = title
        self.mermaid_code = mermaid_code
        self.generation_time = generation_time

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "diagram_type": self.diagram_type,
            "title": self.title,
            "mermaid": self.mermaid_code,
            "generation_time": round(self.generation_time, 3),
        }

    def to_html(self, theme: str = _DEFAULT_MERMAID_THEME) -> str:
        """Wrap Mermaid code in an HTML snippet for frontend rendering.

        Args:
            theme: Mermaid theme name (``default``, ``dark``, ``forest``,
                   ``neutral``).

        Returns:
            An HTML string containing the diagram in a ``<pre class="mermaid">``
            block that can be rendered by the Mermaid.js library.
        """
        return _MERMAID_HTML_TEMPLATE.format(
            diagram_type=self.diagram_type,
            title=self.title,
            mermaid_code=self.mermaid_code,
        )

    def to_markdown(self) -> str:
        """Wrap Mermaid code in a Markdown fenced code block.

        Returns:
            A Markdown string with the diagram in a ``mermaid`` code fence.
        """
        return _MERMAID_MD_TEMPLATE.format(
            title=self.title,
            mermaid_code=self.mermaid_code,
        )


# ---------------------------------------------------------------------------
# Main diagram generator
# ---------------------------------------------------------------------------

class DiagramGenerator:
    """Orchestrate Mermaid.js diagram generation for a generated ML project.

    The generator delegates actual diagram creation to the
    :class:`~backend.agents.diagram_agent.DiagramAgent` and provides
    additional post-processing, validation, and rendering utilities.

    Parameters
    ----------
    validate_mermaid : bool
        If *True* (default), perform lightweight validation on the
        generated Mermaid code to catch obvious syntax issues.
    """

    def __init__(self, validate_mermaid: bool = True) -> None:
        self._agent = DiagramAgent()
        self._validate = validate_mermaid
        logger.info(
            "DiagramGenerator initialised (validate_mermaid={})",
            validate_mermaid,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        research_summary: Dict[str, Any],
        source_files: Dict[str, str] | None = None,
        architecture: Dict[str, Any] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate all three diagram types.

        Args:
            research_summary: Output of ResearchAgent.
            source_files:     Generated source code files (optional but
                              improves accuracy).
            architecture:     Output of ArchitectureAgent (optional).

        Returns:
            A dictionary mapping diagram type names to diagram dicts,
            each containing ``diagram_type``, ``title``, ``mermaid``,
            and ``generation_time`` keys.
        """
        if not research_summary:
            logger.error("No research_summary provided; cannot generate diagrams")
            return {}

        logger.info("Starting diagram generation for {} diagram types", len(DIAGRAM_TYPES))
        start_time = time.monotonic()

        # Build the context for the DiagramAgent
        context: Dict[str, Any] = {
            "research_summary": research_summary,
        }
        if source_files:
            context["files"] = source_files
        if architecture:
            context["architecture"] = architecture

        # Call the DiagramAgent
        try:
            agent_result = await self._agent.run(context)
        except Exception as exc:
            logger.error("DiagramAgent failed: {}", exc)
            return {}

        raw_diagrams: Dict[str, Dict[str, str]] = agent_result.get("diagrams", {})

        if not raw_diagrams:
            logger.warning("DiagramAgent returned no diagrams")
            return {}

        total_elapsed = time.monotonic() - start_time

        # Post-process each diagram
        results: Dict[str, Dict[str, Any]] = {}
        per_diagram_time = total_elapsed / max(len(raw_diagrams), 1)

        for diagram_type, diagram_data in raw_diagrams.items():
            mermaid_code = diagram_data.get("mermaid", "")
            title = diagram_data.get("title", diagram_type.replace("_", " ").title())

            # Clean and validate the Mermaid code
            mermaid_code = self._clean_mermaid(mermaid_code)

            if self._validate:
                is_valid, issue = self._validate_mermaid_syntax(mermaid_code)
                if not is_valid:
                    logger.warning(
                        "Mermaid validation issue for '{}': {}",
                        diagram_type,
                        issue,
                    )
                    # Attempt auto-repair for common issues
                    mermaid_code = self._auto_repair_mermaid(mermaid_code)

            diagram_result = DiagramResult(
                diagram_type=diagram_type,
                title=title,
                mermaid_code=mermaid_code,
                generation_time=per_diagram_time,
            )
            results[diagram_type] = diagram_result.to_dict()

        logger.info(
            "Diagram generation complete: {} diagrams in {:.2f}s",
            len(results),
            total_elapsed,
        )

        return results

    async def generate_single(
        self,
        diagram_type: str,
        research_summary: Dict[str, Any],
        source_files: Dict[str, str] | None = None,
    ) -> Dict[str, Any] | None:
        """Generate a single diagram type.

        This is a convenience method that calls :meth:`generate` and
        extracts the requested diagram type from the result.

        Args:
            diagram_type:     One of ``model_architecture``,
                              ``training_pipeline``, or ``data_flow``.
            research_summary: Output of ResearchAgent.
            source_files:     Generated source code files (optional).

        Returns:
            A diagram dict, or *None* if generation failed.
        """
        if diagram_type not in DIAGRAM_TYPES:
            logger.error(
                "Unknown diagram type '{}'; valid types: {}",
                diagram_type,
                DIAGRAM_TYPES,
            )
            return None

        all_diagrams = await self.generate(
            research_summary=research_summary,
            source_files=source_files,
        )
        return all_diagrams.get(diagram_type)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def wrap_for_frontend(
        diagrams: Dict[str, Dict[str, Any]],
        format: str = "html",
    ) -> Dict[str, str]:
        """Wrap all diagrams in frontend-renderable format.

        Args:
            diagrams: Output of :meth:`generate`.
            format:   Either ``"html"`` or ``"markdown"``.

        Returns:
            A dictionary mapping diagram type to rendered string.
        """
        rendered: Dict[str, str] = {}

        for diagram_type, diagram_data in diagrams.items():
            mermaid_code = diagram_data.get("mermaid", "")
            title = diagram_data.get("title", diagram_type)

            dr = DiagramResult(
                diagram_type=diagram_type,
                title=title,
                mermaid_code=mermaid_code,
            )

            if format == "markdown":
                rendered[diagram_type] = dr.to_markdown()
            else:
                rendered[diagram_type] = dr.to_html()

        return rendered

    @staticmethod
    def get_mermaid_init_script(theme: str = _DEFAULT_MERMAID_THEME) -> str:
        """Return the Mermaid.js initialization script tag.

        This can be included in an HTML page alongside the diagram HTML
        snippets to render them.

        Args:
            theme: Mermaid theme name.

        Returns:
            An HTML ``<script>`` tag that initializes Mermaid.js.
        """
        return textwrap.dedent(f"""\
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
              mermaid.initialize({{
                startOnLoad: true,
                theme: '{theme}',
                securityLevel: 'loose',
                flowchart: {{
                  useMaxWidth: true,
                  htmlLabels: true,
                  curve: 'basis'
                }}
              }});
            </script>
        """)

    # ------------------------------------------------------------------
    # Mermaid cleaning and validation
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_mermaid(raw: str) -> str:
        """Clean raw Mermaid code from LLM output.

        Strips markdown fences, explanatory text, and trailing whitespace.

        Args:
            raw: Raw string from the DiagramAgent.

        Returns:
            Cleaned Mermaid code.
        """
        text = raw.strip()

        # Remove markdown code fences
        text = re.sub(r"^```mermaid\s*\n?", "", text)
        text = re.sub(r"^```\w*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

        # Strip any explanatory text before the first Mermaid keyword
        for keyword in ("flowchart", "graph", "sequenceDiagram", "classDiagram",
                        "stateDiagram", "erDiagram", "gantt", "pie"):
            idx = text.find(keyword)
            if idx > 0:
                text = text[idx:]
                break

        return text.strip()

    @staticmethod
    def _validate_mermaid_syntax(code: str) -> tuple[bool, str]:
        """Perform lightweight validation of Mermaid syntax.

        This does not fully parse Mermaid -- it just checks for the most
        common structural issues that would prevent rendering.

        Args:
            code: Mermaid source code.

        Returns:
            A tuple of (is_valid, issue_description).
        """
        if not code or not code.strip():
            return False, "Empty Mermaid code"

        lines = code.strip().splitlines()
        first_line = lines[0].strip().lower()

        # Must start with a valid Mermaid diagram keyword
        valid_starts = (
            "flowchart", "graph", "sequencediagram", "classdiagram",
            "statediagram", "erdiagram", "gantt", "pie",
        )
        if not any(first_line.startswith(kw) for kw in valid_starts):
            return False, f"Does not start with a valid Mermaid keyword: '{lines[0][:40]}'"

        # Must have at least 2 lines (keyword + at least one node)
        if len(lines) < 2:
            return False, "Diagram has only one line -- missing node definitions"

        # Check for unbalanced brackets (common LLM mistake)
        open_brackets = code.count("[") + code.count("(") + code.count("{")
        close_brackets = code.count("]") + code.count(")") + code.count("}")
        if abs(open_brackets - close_brackets) > 2:
            return False, (
                f"Possibly unbalanced brackets: "
                f"open={open_brackets}, close={close_brackets}"
            )

        # Check for node connections (at least one arrow)
        has_arrow = bool(re.search(r"-->|---|-\.->|==>|--\>|-.->", code))
        if not has_arrow:
            return False, "No node connections (arrows) found"

        return True, ""

    @staticmethod
    def _auto_repair_mermaid(code: str) -> str:
        """Attempt to auto-repair common Mermaid syntax issues.

        Args:
            code: Potentially malformed Mermaid code.

        Returns:
            Repaired Mermaid code (best-effort).
        """
        lines = code.strip().splitlines()

        if not lines:
            return "flowchart TD\n    A[No diagram generated]"

        # Ensure the first line is a valid diagram declaration
        first_line = lines[0].strip()
        if not any(
            first_line.lower().startswith(kw)
            for kw in ("flowchart", "graph", "sequencediagram", "classdiagram")
        ):
            lines.insert(0, "flowchart TD")

        # Fix common issues line by line
        repaired: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Fix unclosed brackets in node labels
            # Count opening vs closing for each bracket type
            for open_ch, close_ch in [("[", "]"), ("(", ")"), ("{", "}")]:
                opens = stripped.count(open_ch)
                closes = stripped.count(close_ch)
                if opens > closes:
                    stripped += close_ch * (opens - closes)

            repaired.append(stripped)

        result = "\n    ".join(repaired)

        # Ensure we have the indent after the first line
        if repaired:
            result = repaired[0] + "\n    " + "\n    ".join(repaired[1:])

        return result
