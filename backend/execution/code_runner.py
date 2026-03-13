"""ReagentAI Code Runner -- Phase 7.

High-level runner that executes a generated ML project inside a Docker
sandbox.  It handles the common two-step workflow:

1. Install Python dependencies from ``requirements.txt``.
2. Execute the main training / model script.

The runner captures stdout and stderr, detects runtime errors, classifies
the error type, and returns a structured :class:`ExecutionResult`.

Usage::

    from backend.execution.code_runner import CodeRunner

    runner = CodeRunner()
    result = await runner.run_project(
        project_path="/abs/path/to/project",
        entry_point="python src/trainer.py",
    )
    if result.success:
        print("Project ran successfully!")
    else:
        print(f"Error ({result.error_type}): {result.error_message}")
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.config.settings import settings
from backend.execution.sandbox import DockerSandbox, SandboxConfig, SandboxResult
from backend.utils.logging import get_logger

logger = get_logger("execution.code_runner")


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class ErrorType:
    """Constants for classifying runtime errors."""

    NONE = "none"
    IMPORT_ERROR = "import_error"
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    RUNTIME_ERROR = "runtime_error"
    VALUE_ERROR = "value_error"
    SHAPE_MISMATCH = "shape_mismatch"
    CUDA_ERROR = "cuda_error"
    MEMORY_ERROR = "memory_error"
    FILE_NOT_FOUND = "file_not_found"
    TIMEOUT = "timeout"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN = "unknown"


# Pattern -> error-type mapping (order matters: first match wins)
_ERROR_PATTERNS: List[tuple[str, str]] = [
    (r"SyntaxError", ErrorType.SYNTAX_ERROR),
    (r"IndentationError", ErrorType.SYNTAX_ERROR),
    (r"ModuleNotFoundError", ErrorType.IMPORT_ERROR),
    (r"ImportError", ErrorType.IMPORT_ERROR),
    (r"No module named", ErrorType.IMPORT_ERROR),
    (r"TypeError", ErrorType.TYPE_ERROR),
    (r"ValueError", ErrorType.VALUE_ERROR),
    (r"RuntimeError.*size mismatch", ErrorType.SHAPE_MISMATCH),
    (r"RuntimeError.*shape", ErrorType.SHAPE_MISMATCH),
    (r"mat1 and mat2 shapes cannot be multiplied", ErrorType.SHAPE_MISMATCH),
    (r"expected .* got .*\bsize\b", ErrorType.SHAPE_MISMATCH),
    (r"CUDA error", ErrorType.CUDA_ERROR),
    (r"CUDA out of memory", ErrorType.MEMORY_ERROR),
    (r"MemoryError", ErrorType.MEMORY_ERROR),
    (r"OutOfMemoryError", ErrorType.MEMORY_ERROR),
    (r"Killed", ErrorType.MEMORY_ERROR),
    (r"FileNotFoundError", ErrorType.FILE_NOT_FOUND),
    (r"No such file or directory", ErrorType.FILE_NOT_FOUND),
    (r"RuntimeError", ErrorType.RUNTIME_ERROR),
    (r"Exception", ErrorType.RUNTIME_ERROR),
    (r"ERROR: Could not find a version that satisfies", ErrorType.DEPENDENCY_ERROR),
    (r"ERROR: No matching distribution found", ErrorType.DEPENDENCY_ERROR),
    (r"pip.*error", ErrorType.DEPENDENCY_ERROR),
]


def classify_error(stderr: str) -> str:
    """Classify stderr text into an error type constant.

    Scans *stderr* against known patterns and returns the first matching
    :class:`ErrorType` constant, or ``ErrorType.UNKNOWN`` if no pattern
    matches.

    Args:
        stderr: The raw standard error output.

    Returns:
        An error type string from :class:`ErrorType`.
    """
    for pattern, error_type in _ERROR_PATTERNS:
        if re.search(pattern, stderr, re.IGNORECASE):
            return error_type
    return ErrorType.UNKNOWN


def extract_error_message(stderr: str) -> str:
    """Extract a concise error message from a Python traceback.

    Looks for the last exception line in the traceback, which is typically
    the most informative.  Falls back to the last non-empty line of stderr.

    Args:
        stderr: The raw standard error output.

    Returns:
        A single-line error message.
    """
    if not stderr.strip():
        return ""

    # Try to find the final exception line (e.g. "ValueError: bad value")
    exception_lines = re.findall(
        r"^(\w*Error\b.*|^\w*Exception\b.*)$", stderr, re.MULTILINE
    )
    if exception_lines:
        return exception_lines[-1].strip()[:500]

    # Fallback: last non-empty line
    lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
    if lines:
        return lines[-1][:500]

    return stderr.strip()[:500]


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Structured result of running a generated project.

    Attributes:
        success:         ``True`` if the project ran without errors.
        stdout:          Captured standard output.
        stderr:          Captured standard error.
        error_type:      Classified error type (see :class:`ErrorType`).
        error_message:   Concise human-readable error description.
        execution_time:  Wall-clock seconds for the entire run (install + exec).
        install_stdout:  Stdout from the dependency installation step.
        install_stderr:  Stderr from the dependency installation step.
        install_success: ``True`` if dependency installation succeeded.
        exit_code:       Process exit code from the main execution.
        timed_out:       ``True`` if execution exceeded the timeout.
    """

    success: bool = False
    stdout: str = ""
    stderr: str = ""
    error_type: str = ErrorType.NONE
    error_message: str = ""
    execution_time: float = 0.0
    install_stdout: str = ""
    install_stderr: str = ""
    install_success: bool = False
    exit_code: int = -1
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Code Runner
# ---------------------------------------------------------------------------

class CodeRunner:
    """Execute generated ML projects inside a Docker sandbox.

    The runner performs two sequential steps:

    1. **Install dependencies** -- runs ``pip install -r requirements.txt``
       (if the file exists) inside the sandbox.
    2. **Execute the entry point** -- runs the main script (e.g.
       ``python src/trainer.py``) and captures the outcome.

    Both steps share the same sandbox configuration but use separate
    container invocations to isolate failures.
    """

    def __init__(
        self,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        """Initialise the code runner.

        Args:
            sandbox_config: Optional sandbox configuration override.
        """
        self._sandbox = DockerSandbox(config=sandbox_config)
        logger.info("CodeRunner initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_project(
        self,
        project_path: str,
        *,
        entry_point: str = "python src/trainer.py",
        install_deps: bool = True,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Run a generated project inside the sandbox.

        Args:
            project_path: Absolute host path to the generated project.
            entry_point:  Shell command to run the main script.
            install_deps: Whether to install requirements.txt first.
            timeout:      Override the sandbox timeout (seconds).

        Returns:
            An :class:`ExecutionResult` with full execution details.
        """
        start_time = time.monotonic()
        logger.info(
            "Running project at '{}' (entry_point='{}', install_deps={})",
            project_path,
            entry_point,
            install_deps,
        )

        result = ExecutionResult()

        # Step 1: Install dependencies
        if install_deps:
            install_result = await self._install_dependencies(
                project_path, timeout=timeout
            )
            result.install_stdout = install_result.stdout
            result.install_stderr = install_result.stderr
            result.install_success = install_result.exit_code == 0

            if not result.install_success:
                logger.warning(
                    "Dependency installation failed (exit_code={})",
                    install_result.exit_code,
                )
                # Check if this is a hard failure (syntax error in requirements,
                # missing package) vs. a soft warning
                if self._is_hard_install_failure(install_result):
                    result.success = False
                    result.stderr = install_result.stderr
                    result.error_type = classify_error(install_result.stderr)
                    result.error_message = extract_error_message(
                        install_result.stderr
                    )
                    result.exit_code = install_result.exit_code
                    result.execution_time = time.monotonic() - start_time
                    logger.error(
                        "Hard install failure -- aborting execution: {}",
                        result.error_message,
                    )
                    return result
                else:
                    logger.info(
                        "Install had warnings but proceeding with execution"
                    )
        else:
            result.install_success = True

        # Step 2: Execute the entry point
        exec_result = await self._execute_entry_point(
            project_path, entry_point, timeout=timeout
        )

        result.stdout = exec_result.stdout
        result.stderr = exec_result.stderr
        result.exit_code = exec_result.exit_code
        result.timed_out = exec_result.timed_out
        result.execution_time = time.monotonic() - start_time

        if exec_result.timed_out:
            result.success = False
            result.error_type = ErrorType.TIMEOUT
            result.error_message = (
                f"Execution timed out after {exec_result.execution_time:.0f}s"
            )
            logger.warning("Execution timed out")
        elif exec_result.exit_code == 0:
            result.success = True
            result.error_type = ErrorType.NONE
            result.error_message = ""
            logger.info(
                "Project executed successfully in {:.1f}s",
                result.execution_time,
            )
        else:
            result.success = False
            result.error_type = classify_error(exec_result.stderr)
            result.error_message = extract_error_message(exec_result.stderr)
            logger.warning(
                "Execution failed (exit_code={}, error_type={}): {}",
                exec_result.exit_code,
                result.error_type,
                result.error_message[:200],
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _install_dependencies(
        self,
        project_path: str,
        *,
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """Install Python dependencies from requirements.txt.

        Uses ``pip install --no-cache-dir -r requirements.txt`` inside
        the sandbox.  If ``requirements.txt`` does not exist, the command
        is a no-op that exits successfully.

        Args:
            project_path: Absolute host path to the project.
            timeout:      Optional timeout override.

        Returns:
            A :class:`SandboxResult` from the installation command.
        """
        # The command checks for requirements.txt first to avoid a hard
        # failure when the file is missing.
        install_cmd = (
            "if [ -f requirements.txt ]; then "
            "pip install --no-cache-dir -r requirements.txt; "
            "else echo 'No requirements.txt found -- skipping install'; fi"
        )

        logger.info("Installing dependencies for '{}'", project_path)

        # Allow extra time for pip to download packages
        install_timeout = (timeout or self._sandbox._config.timeout) + 120

        return await self._sandbox.run_in_sandbox(
            project_path=project_path,
            command=install_cmd,
            timeout=install_timeout,
        )

    async def _execute_entry_point(
        self,
        project_path: str,
        entry_point: str,
        *,
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """Execute the project entry-point script in the sandbox.

        Args:
            project_path: Absolute host path to the project.
            entry_point:  Shell command to run.
            timeout:      Optional timeout override.

        Returns:
            A :class:`SandboxResult` from the execution.
        """
        logger.info(
            "Executing entry point '{}' for '{}'", entry_point, project_path
        )
        return await self._sandbox.run_in_sandbox(
            project_path=project_path,
            command=entry_point,
            timeout=timeout,
        )

    @staticmethod
    def _is_hard_install_failure(result: SandboxResult) -> bool:
        """Determine whether a pip install failure is unrecoverable.

        Soft failures (e.g. pip upgrade warnings, optional dependency
        warnings) should not block execution.

        Args:
            result: The sandbox result from the install step.

        Returns:
            ``True`` if the failure is a hard blocker.
        """
        stderr = result.stderr.lower()

        # Hard failure indicators
        hard_indicators = [
            "could not find a version that satisfies",
            "no matching distribution found",
            "syntaxerror",
            "error: subprocess-exited-with-error",
            "failed building wheel",
            "modulenotfounderror",
        ]

        for indicator in hard_indicators:
            if indicator in stderr:
                return True

        # If exit code is non-zero but stderr only has warnings, treat as soft
        if result.exit_code != 0:
            # Check if there are actual error lines (not just warnings)
            error_lines = [
                line
                for line in result.stderr.splitlines()
                if "error" in line.lower() and "warning" not in line.lower()
            ]
            return len(error_lines) > 0

        return False

    def close(self) -> None:
        """Release sandbox resources."""
        self._sandbox.close()
