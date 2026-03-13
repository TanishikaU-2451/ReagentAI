"""ReagentAI Code Generation Pipeline -- Phase 6.

Orchestrates the full code-generation process:

1. Receives outputs from the Research, Planning, and Architecture agents
   (and optionally reference code retrieved from GitHub).
2. Invokes the :class:`CodingAgent` to generate every file listed in the
   architecture specification.
3. Persists the generated files to disk under
   ``generated_projects/{project_id}/``, creating the proper directory
   structure.
4. Returns a manifest of all generated file paths.

Usage::

    from backend.orchestration.code_generator import CodeGenerationPipeline

    pipeline = CodeGenerationPipeline()
    result = await pipeline.generate(
        project_id="abc123",
        plan={...},
        architecture={...},
        research_summary={...},
        reference_code={"model.py": "..."},
    )
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.agents import CodingAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger

logger = get_logger("orchestration.code_generator")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeneratedFile:
    """Metadata for a single generated file."""

    relative_path: str
    absolute_path: str
    size_bytes: int
    status: str  # "ok" or "error"
    error: Optional[str] = None


@dataclass
class CodeGenerationResult:
    """Result returned by the code generation pipeline.

    Attributes:
        project_id:       Unique identifier for this generation run.
        project_dir:      Absolute path to the generated project directory.
        files:            Dict mapping relative paths to their source code.
        file_manifest:    List of :class:`GeneratedFile` metadata entries.
        generation_log:   Raw generation log from the CodingAgent.
        elapsed_seconds:  Wall-clock time for the entire generation.
        success:          ``True`` if at least one file was generated.
    """

    project_id: str
    project_dir: str
    files: Dict[str, str] = field(default_factory=dict)
    file_manifest: List[GeneratedFile] = field(default_factory=list)
    generation_log: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    success: bool = False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CodeGenerationPipeline:
    """Orchestrate the CodingAgent to produce a complete project on disk.

    The pipeline:

    * Merges reference code (from GitHub retrieval) into the architecture
      context so the CodingAgent can use it as inspiration.
    * Delegates actual file generation to the CodingAgent.
    * Writes every generated file to the filesystem under the configured
      ``generated_projects_path``.
    * Returns a :class:`CodeGenerationResult` with full metadata.
    """

    def __init__(self) -> None:
        self._coding_agent = CodingAgent()
        self._base_dir: Path = Path(settings.generated_projects_path).resolve()
        logger.info(
            "CodeGenerationPipeline initialised (output dir: {})",
            self._base_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        project_id: Optional[str] = None,
        *,
        plan: Optional[Dict[str, Any]] = None,
        architecture: Optional[Dict[str, Any]] = None,
        research_summary: Optional[Dict[str, Any]] = None,
        reference_code: Optional[Dict[str, str]] = None,
    ) -> CodeGenerationResult:
        """Run the full code-generation pipeline.

        Args:
            project_id:       Optional unique project identifier.  A UUID is
                              generated when omitted.
            plan:             Output of the PlanningAgent.
            architecture:     Output of the ArchitectureAgent.
            research_summary: Output of the ResearchAgent.
            reference_code:   Optional dict of reference source files
                              retrieved from GitHub (path -> code).

        Returns:
            A :class:`CodeGenerationResult` containing the generated files,
            their disk locations, and generation metadata.
        """
        start_time = time.monotonic()

        # Resolve project identity and output directory
        project_id = project_id or uuid.uuid4().hex[:12]
        project_dir = self._base_dir / project_id
        logger.info("Starting code generation for project '{}'", project_id)

        # Ensure default dicts
        plan = plan or {}
        architecture = architecture or {}
        research_summary = research_summary or {}

        # Augment architecture context with reference code if available
        augmented_architecture = self._augment_with_reference_code(
            architecture, reference_code
        )

        # Build the CodingAgent context
        coding_context: Dict[str, Any] = {
            "research_summary": research_summary,
            "plan": plan,
            "architecture": augmented_architecture,
        }

        # Invoke the CodingAgent
        logger.info("Invoking CodingAgent ...")
        try:
            agent_result = await self._coding_agent.run(coding_context)
        except Exception as exc:
            logger.error("CodingAgent raised an exception: {}", exc)
            return CodeGenerationResult(
                project_id=project_id,
                project_dir=str(project_dir),
                elapsed_seconds=time.monotonic() - start_time,
                success=False,
            )

        generated_files: Dict[str, str] = agent_result.get("files", {})
        generation_log: List[Dict[str, Any]] = agent_result.get(
            "generation_log", []
        )

        if not generated_files:
            logger.warning(
                "CodingAgent returned no files for project '{}'", project_id
            )
            return CodeGenerationResult(
                project_id=project_id,
                project_dir=str(project_dir),
                generation_log=generation_log,
                elapsed_seconds=time.monotonic() - start_time,
                success=False,
            )

        # Write files to disk
        file_manifest = await self._write_files_to_disk(
            project_dir, generated_files
        )

        elapsed = time.monotonic() - start_time
        ok_count = sum(1 for f in file_manifest if f.status == "ok")

        logger.info(
            "Code generation complete for '{}': {}/{} files written in {:.1f}s",
            project_id,
            ok_count,
            len(generated_files),
            elapsed,
        )

        return CodeGenerationResult(
            project_id=project_id,
            project_dir=str(project_dir),
            files=generated_files,
            file_manifest=file_manifest,
            generation_log=generation_log,
            elapsed_seconds=elapsed,
            success=ok_count > 0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _augment_with_reference_code(
        architecture: Dict[str, Any],
        reference_code: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Merge reference code snippets into the architecture context.

        The reference code is stored under the ``reference_code`` key so
        the CodingAgent's shared-context builder can include it in the
        prompt without overwriting the original architecture data.

        Args:
            architecture:   The original architecture dict.
            reference_code: Optional dict mapping file names to code.

        Returns:
            A (shallow) copy of *architecture* with reference code injected.
        """
        if not reference_code:
            return architecture

        augmented = dict(architecture)

        # Build a compact summary of each reference file (first 80 lines)
        summaries: Dict[str, str] = {}
        for path, code in reference_code.items():
            snippet_lines = code.splitlines()[:80]
            summaries[path] = "\n".join(snippet_lines)

        augmented["reference_code"] = summaries
        logger.info(
            "Augmented architecture with {} reference code files",
            len(summaries),
        )
        return augmented

    async def _write_files_to_disk(
        self,
        project_dir: Path,
        files: Dict[str, str],
    ) -> List[GeneratedFile]:
        """Persist generated files to the filesystem.

        Creates the full directory tree under *project_dir* and writes each
        file.  File I/O is offloaded to the default executor so it does not
        block the event loop.

        Args:
            project_dir: Root directory for the generated project.
            files:       Dict mapping relative file paths to their content.

        Returns:
            A list of :class:`GeneratedFile` metadata entries.
        """
        manifest: List[GeneratedFile] = []
        loop = asyncio.get_running_loop()

        for relative_path, content in files.items():
            abs_path = project_dir / relative_path

            try:
                # Create parent directories in executor
                await loop.run_in_executor(
                    None,
                    lambda p=abs_path: p.parent.mkdir(parents=True, exist_ok=True),
                )

                # Write file content in executor
                encoded = content.encode("utf-8")
                await loop.run_in_executor(
                    None,
                    lambda p=abs_path, c=encoded: p.write_bytes(c),
                )

                manifest.append(
                    GeneratedFile(
                        relative_path=relative_path,
                        absolute_path=str(abs_path),
                        size_bytes=len(encoded),
                        status="ok",
                    )
                )
                logger.debug("Wrote {} ({} bytes)", relative_path, len(encoded))

            except OSError as exc:
                logger.error("Failed to write {}: {}", relative_path, exc)
                manifest.append(
                    GeneratedFile(
                        relative_path=relative_path,
                        absolute_path=str(abs_path),
                        size_bytes=0,
                        status="error",
                        error=str(exc),
                    )
                )

        return manifest

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def regenerate_files(
        self,
        project_id: str,
        files_to_regenerate: Dict[str, str],
        *,
        plan: Optional[Dict[str, Any]] = None,
        architecture: Optional[Dict[str, Any]] = None,
        research_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Re-generate (or replace) specific files in an existing project.

        This is used by the debug loop to patch individual files after the
        DebugAgent produces corrected source code.

        Args:
            project_id:           The existing project identifier.
            files_to_regenerate:  Dict mapping relative paths to their new
                                  source code content.
            plan:                 PlanningAgent output (for context).
            architecture:         ArchitectureAgent output (for context).
            research_summary:     ResearchAgent output (for context).

        Returns:
            Dict of relative paths to their updated code, reflecting what
            was written to disk.
        """
        project_dir = self._base_dir / project_id
        logger.info(
            "Regenerating {} files for project '{}'",
            len(files_to_regenerate),
            project_id,
        )

        updated_files: Dict[str, str] = {}
        loop = asyncio.get_running_loop()

        for relative_path, new_code in files_to_regenerate.items():
            abs_path = project_dir / relative_path
            try:
                await loop.run_in_executor(
                    None,
                    lambda p=abs_path: p.parent.mkdir(parents=True, exist_ok=True),
                )
                encoded = new_code.encode("utf-8")
                await loop.run_in_executor(
                    None,
                    lambda p=abs_path, c=encoded: p.write_bytes(c),
                )
                updated_files[relative_path] = new_code
                logger.debug(
                    "Regenerated {} ({} bytes)", relative_path, len(encoded)
                )
            except OSError as exc:
                logger.error(
                    "Failed to regenerate {}: {}", relative_path, exc
                )

        return updated_files

    def get_project_dir(self, project_id: str) -> Path:
        """Return the absolute path for a given project id.

        Args:
            project_id: The project identifier.

        Returns:
            The resolved :class:`Path` to the project directory.
        """
        return self._base_dir / project_id
