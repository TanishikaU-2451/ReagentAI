"""ReagentAI Self-Debug Loop -- Phase 8.

Implements the iterative generate-execute-debug pipeline:

1. **Generate** code via :class:`CodeGenerationPipeline`.
2. **Execute** the generated project in a Docker sandbox via
   :class:`CodeRunner`.
3. **Capture** any runtime errors from the execution result.
4. **Diagnose and fix** errors via the :class:`DebugAgent`.
5. **Patch** the project files with the corrected code.
6. **Repeat** from step 2 until the project runs successfully or the
   maximum number of attempts is exhausted.

Every iteration is recorded so that callers can inspect the full history
of attempts, errors encountered, and fixes applied.

Usage::

    from backend.orchestration.debug_loop import DebugLoop

    loop = DebugLoop()
    result = await loop.run(
        project_id="abc123",
        plan={...},
        architecture={...},
        research_summary={...},
    )

    if result.final_success:
        print("Project works!")
    else:
        print(f"Failed after {result.attempts_made} attempts")
        for it in result.iteration_history:
            print(f"  Attempt {it['attempt']}: {it['error_type']}")
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.agents import DebugAgent
from backend.config.settings import settings
from backend.execution.code_runner import CodeRunner, ExecutionResult, ErrorType
from backend.orchestration.code_generator import CodeGenerationPipeline
from backend.utils.logging import get_logger

logger = get_logger("orchestration.debug_loop")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Record of a single debug-loop iteration.

    Attributes:
        attempt:         1-based attempt number.
        execution_result: The :class:`ExecutionResult` from this attempt.
        error_type:      Classified error type (or ``"none"`` on success).
        error_message:   Human-readable error description.
        diagnosis:       Root-cause explanation from the DebugAgent.
        fix_applied:     Summary of changes applied by the DebugAgent.
        corrected_files: List of file paths that were modified.
        confidence:      DebugAgent's confidence in the fix (0.0 -- 1.0).
        elapsed_seconds: Wall-clock time for this iteration.
    """

    attempt: int = 0
    execution_result: Optional[ExecutionResult] = None
    error_type: str = ErrorType.NONE
    error_message: str = ""
    diagnosis: str = ""
    fix_applied: str = ""
    corrected_files: List[str] = field(default_factory=list)
    confidence: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this record to a plain dictionary.

        Returns:
            A JSON-serialisable dictionary representation.
        """
        return {
            "attempt": self.attempt,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "diagnosis": self.diagnosis,
            "fix_applied": self.fix_applied,
            "corrected_files": self.corrected_files,
            "confidence": self.confidence,
            "elapsed_seconds": self.elapsed_seconds,
            "success": (
                self.execution_result.success
                if self.execution_result
                else False
            ),
        }


@dataclass
class DebugLoopResult:
    """Final result of the complete debug loop.

    Attributes:
        final_success:     ``True`` if the project ran successfully after
                           all iterations.
        attempts_made:     Total number of execution attempts.
        iteration_history: Ordered list of :class:`IterationRecord` entries.
        final_code:        Dict mapping relative file paths to their final
                           source code content.
        project_id:        The project identifier.
        project_dir:       Absolute path to the project directory.
        total_elapsed:     Wall-clock seconds for the entire debug loop.
    """

    final_success: bool = False
    attempts_made: int = 0
    iteration_history: List[IterationRecord] = field(default_factory=list)
    final_code: Dict[str, str] = field(default_factory=dict)
    project_id: str = ""
    project_dir: str = ""
    total_elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary.

        Returns:
            A JSON-serialisable dictionary representation.
        """
        return {
            "final_success": self.final_success,
            "attempts_made": self.attempts_made,
            "iteration_history": [it.to_dict() for it in self.iteration_history],
            "project_id": self.project_id,
            "project_dir": self.project_dir,
            "total_elapsed": self.total_elapsed,
            "final_file_count": len(self.final_code),
        }


# ---------------------------------------------------------------------------
# Debug loop
# ---------------------------------------------------------------------------

class DebugLoop:
    """Orchestrate the iterative generate-execute-debug cycle.

    The loop coordinates three components:

    * :class:`CodeGenerationPipeline` -- for the initial code generation
      and for writing corrected files back to disk.
    * :class:`CodeRunner` -- for executing the project in a Docker sandbox.
    * :class:`DebugAgent` -- for diagnosing errors and producing fixes.

    The maximum number of retry attempts is controlled by
    ``settings.max_debug_attempts``.
    """

    def __init__(
        self,
        *,
        max_attempts: Optional[int] = None,
        code_generator: Optional[CodeGenerationPipeline] = None,
        code_runner: Optional[CodeRunner] = None,
        debug_agent: Optional[DebugAgent] = None,
    ) -> None:
        """Initialise the debug loop.

        Args:
            max_attempts:   Maximum number of execute-debug iterations.
                            Defaults to ``settings.max_debug_attempts``.
            code_generator: Optional pre-configured code generation pipeline.
            code_runner:    Optional pre-configured code runner.
            debug_agent:    Optional pre-configured debug agent.
        """
        self._max_attempts = max_attempts or settings.max_debug_attempts
        self._generator = code_generator or CodeGenerationPipeline()
        self._runner = code_runner or CodeRunner()
        self._debug_agent = debug_agent or DebugAgent()

        logger.info(
            "DebugLoop initialised (max_attempts={})", self._max_attempts
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        project_id: Optional[str] = None,
        *,
        plan: Optional[Dict[str, Any]] = None,
        architecture: Optional[Dict[str, Any]] = None,
        research_summary: Optional[Dict[str, Any]] = None,
        reference_code: Optional[Dict[str, str]] = None,
        entry_point: str = "python src/trainer.py",
    ) -> DebugLoopResult:
        """Execute the full generate-execute-debug loop.

        Generates the project, then iteratively executes and debugs it
        until it succeeds or the attempt budget is exhausted.

        Args:
            project_id:       Optional project identifier (auto-generated
                              if omitted).
            plan:             PlanningAgent output.
            architecture:     ArchitectureAgent output.
            research_summary: ResearchAgent output.
            reference_code:   Optional GitHub reference code.
            entry_point:      Shell command to execute the project.

        Returns:
            A :class:`DebugLoopResult` with the final outcome and full
            iteration history.
        """
        total_start = time.monotonic()

        logger.info(
            "=== Debug loop starting (max_attempts={}) ===",
            self._max_attempts,
        )

        # Phase 6: Initial code generation
        logger.info("Phase 6: Generating initial code ...")
        gen_result = await self._generator.generate(
            project_id=project_id,
            plan=plan,
            architecture=architecture,
            research_summary=research_summary,
            reference_code=reference_code,
        )

        if not gen_result.success:
            logger.error("Initial code generation failed -- aborting loop")
            return DebugLoopResult(
                final_success=False,
                attempts_made=0,
                project_id=gen_result.project_id,
                project_dir=gen_result.project_dir,
                total_elapsed=time.monotonic() - total_start,
            )

        project_id = gen_result.project_id
        project_dir = gen_result.project_dir
        current_files = dict(gen_result.files)
        iteration_history: List[IterationRecord] = []
        previous_fixes: List[str] = []

        logger.info(
            "Initial generation complete: {} files in '{}'",
            len(current_files),
            project_dir,
        )

        # Execute-debug loop
        for attempt in range(1, self._max_attempts + 1):
            iter_start = time.monotonic()

            logger.info(
                "--- Attempt {}/{} ---", attempt, self._max_attempts
            )

            # Execute the project
            exec_result = await self._runner.run_project(
                project_path=project_dir,
                entry_point=entry_point,
            )

            record = IterationRecord(
                attempt=attempt,
                execution_result=exec_result,
                error_type=exec_result.error_type,
                error_message=exec_result.error_message,
            )

            # Success -- we are done
            if exec_result.success:
                record.elapsed_seconds = time.monotonic() - iter_start
                iteration_history.append(record)

                logger.info(
                    "Project executed successfully on attempt {}", attempt
                )

                return DebugLoopResult(
                    final_success=True,
                    attempts_made=attempt,
                    iteration_history=iteration_history,
                    final_code=current_files,
                    project_id=project_id,
                    project_dir=project_dir,
                    total_elapsed=time.monotonic() - total_start,
                )

            # Failure -- attempt to debug (unless this is the last attempt)
            logger.warning(
                "Attempt {} failed: [{}] {}",
                attempt,
                exec_result.error_type,
                exec_result.error_message[:200],
            )

            if attempt >= self._max_attempts:
                record.elapsed_seconds = time.monotonic() - iter_start
                iteration_history.append(record)
                logger.error(
                    "Max attempts ({}) reached -- giving up",
                    self._max_attempts,
                )
                break

            # Invoke the DebugAgent
            debug_context = self._build_debug_context(
                exec_result=exec_result,
                current_files=current_files,
                attempt=attempt,
                previous_fixes=previous_fixes,
                research_summary=research_summary,
            )

            logger.info("Invoking DebugAgent (attempt {}) ...", attempt)
            try:
                debug_result = await self._debug_agent.run(debug_context)
            except Exception as exc:
                logger.error("DebugAgent raised an exception: {}", exc)
                record.diagnosis = f"DebugAgent error: {exc}"
                record.elapsed_seconds = time.monotonic() - iter_start
                iteration_history.append(record)
                continue

            # Extract the fix
            diagnosis = debug_result.get("diagnosis", "No diagnosis")
            corrected_files = debug_result.get("corrected_files", {})
            changes_summary = debug_result.get("changes_summary", [])
            confidence = debug_result.get("confidence", 0.0)

            record.diagnosis = diagnosis
            record.fix_applied = "; ".join(changes_summary) if changes_summary else diagnosis
            record.corrected_files = list(corrected_files.keys())
            record.confidence = confidence

            logger.info(
                "DebugAgent diagnosis: {} (confidence={:.2f}, files_fixed={})",
                diagnosis[:150],
                confidence,
                len(corrected_files),
            )

            if not corrected_files:
                logger.warning(
                    "DebugAgent returned no corrected files -- retrying"
                )
                record.elapsed_seconds = time.monotonic() - iter_start
                iteration_history.append(record)
                previous_fixes.append(
                    f"Attempt {attempt}: {diagnosis} (no files corrected)"
                )
                continue

            # Apply the fix: update in-memory files and write to disk
            current_files = self._apply_corrections(
                current_files, corrected_files
            )

            await self._generator.regenerate_files(
                project_id=project_id,
                files_to_regenerate=corrected_files,
            )

            logger.info(
                "Applied fixes to {} files: {}",
                len(corrected_files),
                list(corrected_files.keys()),
            )

            # Record fix summary for future attempts
            fix_summary = (
                f"Attempt {attempt}: {diagnosis[:200]} -- "
                f"fixed {list(corrected_files.keys())}"
            )
            previous_fixes.append(fix_summary)

            record.elapsed_seconds = time.monotonic() - iter_start
            iteration_history.append(record)

        # Loop exhausted without success
        total_elapsed = time.monotonic() - total_start

        logger.error(
            "Debug loop completed without success after {} attempts "
            "({:.1f}s total)",
            len(iteration_history),
            total_elapsed,
        )

        return DebugLoopResult(
            final_success=False,
            attempts_made=len(iteration_history),
            iteration_history=iteration_history,
            final_code=current_files,
            project_id=project_id,
            project_dir=project_dir,
            total_elapsed=total_elapsed,
        )

    # ------------------------------------------------------------------
    # Run with pre-generated code
    # ------------------------------------------------------------------

    async def run_with_existing_code(
        self,
        project_id: str,
        project_dir: str,
        files: Dict[str, str],
        *,
        research_summary: Optional[Dict[str, Any]] = None,
        entry_point: str = "python src/trainer.py",
    ) -> DebugLoopResult:
        """Run the debug loop on an already-generated project.

        Skips the initial code generation step and proceeds directly to
        the execute-debug cycle.  Useful when code has already been
        generated and only the debug loop needs to be (re-)run.

        Args:
            project_id:       The project identifier.
            project_dir:      Absolute path to the project directory.
            files:            Dict of current source files (path -> code).
            research_summary: Optional research context for the DebugAgent.
            entry_point:      Shell command to execute.

        Returns:
            A :class:`DebugLoopResult`.
        """
        total_start = time.monotonic()
        current_files = dict(files)
        iteration_history: List[IterationRecord] = []
        previous_fixes: List[str] = []

        logger.info(
            "Debug loop starting on existing project '{}' ({} files)",
            project_id,
            len(current_files),
        )

        for attempt in range(1, self._max_attempts + 1):
            iter_start = time.monotonic()

            logger.info(
                "--- Attempt {}/{} ---", attempt, self._max_attempts
            )

            exec_result = await self._runner.run_project(
                project_path=project_dir,
                entry_point=entry_point,
            )

            record = IterationRecord(
                attempt=attempt,
                execution_result=exec_result,
                error_type=exec_result.error_type,
                error_message=exec_result.error_message,
            )

            if exec_result.success:
                record.elapsed_seconds = time.monotonic() - iter_start
                iteration_history.append(record)

                logger.info(
                    "Project succeeded on attempt {}", attempt
                )

                return DebugLoopResult(
                    final_success=True,
                    attempts_made=attempt,
                    iteration_history=iteration_history,
                    final_code=current_files,
                    project_id=project_id,
                    project_dir=project_dir,
                    total_elapsed=time.monotonic() - total_start,
                )

            logger.warning(
                "Attempt {} failed: [{}] {}",
                attempt,
                exec_result.error_type,
                exec_result.error_message[:200],
            )

            if attempt >= self._max_attempts:
                record.elapsed_seconds = time.monotonic() - iter_start
                iteration_history.append(record)
                break

            # Debug
            debug_context = self._build_debug_context(
                exec_result=exec_result,
                current_files=current_files,
                attempt=attempt,
                previous_fixes=previous_fixes,
                research_summary=research_summary,
            )

            try:
                debug_result = await self._debug_agent.run(debug_context)
            except Exception as exc:
                logger.error("DebugAgent error: {}", exc)
                record.diagnosis = f"DebugAgent error: {exc}"
                record.elapsed_seconds = time.monotonic() - iter_start
                iteration_history.append(record)
                continue

            diagnosis = debug_result.get("diagnosis", "No diagnosis")
            corrected_files = debug_result.get("corrected_files", {})
            changes_summary = debug_result.get("changes_summary", [])
            confidence = debug_result.get("confidence", 0.0)

            record.diagnosis = diagnosis
            record.fix_applied = "; ".join(changes_summary) if changes_summary else diagnosis
            record.corrected_files = list(corrected_files.keys())
            record.confidence = confidence

            if not corrected_files:
                record.elapsed_seconds = time.monotonic() - iter_start
                iteration_history.append(record)
                previous_fixes.append(
                    f"Attempt {attempt}: {diagnosis} (no files corrected)"
                )
                continue

            current_files = self._apply_corrections(
                current_files, corrected_files
            )

            await self._generator.regenerate_files(
                project_id=project_id,
                files_to_regenerate=corrected_files,
            )

            previous_fixes.append(
                f"Attempt {attempt}: {diagnosis[:200]} -- "
                f"fixed {list(corrected_files.keys())}"
            )

            record.elapsed_seconds = time.monotonic() - iter_start
            iteration_history.append(record)

        total_elapsed = time.monotonic() - total_start

        return DebugLoopResult(
            final_success=False,
            attempts_made=len(iteration_history),
            iteration_history=iteration_history,
            final_code=current_files,
            project_id=project_id,
            project_dir=project_dir,
            total_elapsed=total_elapsed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_debug_context(
        exec_result: ExecutionResult,
        current_files: Dict[str, str],
        attempt: int,
        previous_fixes: List[str],
        research_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the context dictionary for the DebugAgent.

        Combines the error output, current source files, attempt number,
        history of previous fixes, and optional research context into the
        format expected by :meth:`DebugAgent.run`.

        Args:
            exec_result:      The execution result containing error output.
            current_files:    Current state of all project source files.
            attempt:          Current 1-based attempt number.
            previous_fixes:   Summaries of prior fix attempts.
            research_summary: Optional research context.

        Returns:
            A context dictionary ready for the DebugAgent.
        """
        # Combine stderr with any relevant stdout hints
        error_output = exec_result.stderr
        if not error_output.strip() and exec_result.stdout:
            # Sometimes errors are printed to stdout
            error_output = exec_result.stdout

        # If install failed, prepend install stderr
        if not exec_result.install_success and exec_result.install_stderr:
            error_output = (
                "=== DEPENDENCY INSTALLATION ERRORS ===\n"
                + exec_result.install_stderr
                + "\n\n=== EXECUTION ERRORS ===\n"
                + error_output
            )

        context: Dict[str, Any] = {
            "error_output": error_output,
            "files": current_files,
            "attempt": attempt,
            "previous_fixes": previous_fixes,
        }

        if research_summary:
            context["research_summary"] = research_summary

        return context

    @staticmethod
    def _apply_corrections(
        current_files: Dict[str, str],
        corrected_files: Dict[str, str],
    ) -> Dict[str, str]:
        """Merge corrected files into the current file set.

        Creates a new dictionary so the original is not mutated
        (preserving the snapshot for iteration history).

        Args:
            current_files:  The current state of all project files.
            corrected_files: Files corrected by the DebugAgent.

        Returns:
            A new dict with the corrections applied.
        """
        updated = dict(current_files)
        for filepath, new_code in corrected_files.items():
            if filepath in updated:
                logger.debug(
                    "Patching existing file: {} ({} -> {} chars)",
                    filepath,
                    len(updated[filepath]),
                    len(new_code),
                )
            else:
                logger.debug(
                    "Adding new file from debug fix: {} ({} chars)",
                    filepath,
                    len(new_code),
                )
            updated[filepath] = new_code
        return updated
