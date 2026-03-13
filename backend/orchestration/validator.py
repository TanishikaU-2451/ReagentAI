"""ReagentAI Implementation Validation Pipeline -- Phase 9.

Orchestrates lightweight validation of a generated ML project by delegating
static checks to the :class:`~backend.agents.validation_agent.ValidationAgent`
and executing the generated validation scripts inside the Docker sandbox.

The pipeline performs the following checks:

1. **Model forward pass** -- Instantiates the model with dummy input and
   verifies that ``forward()`` returns a tensor of the expected shape.
2. **Dataset loading** -- Instantiates the dataset loader and verifies that
   ``__len__`` and ``__getitem__`` work without errors.
3. **Training loop initialization** -- Constructs the trainer with the model,
   dataset, and config and verifies that one optimiser step can execute.

All checks are run through small, self-contained Python scripts executed
inside the execution sandbox so the host environment is never polluted.

Usage::

    from backend.orchestration.validator import ValidationPipeline

    pipeline = ValidationPipeline()
    report = await pipeline.validate(
        source_files={"src/model.py": "...", ...},
        research_summary={...},
    )
    print(report["overall_status"])  # "passed" | "failed" | "partial"
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.agents import ValidationAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger

logger = get_logger("orchestration.validator")

# ---------------------------------------------------------------------------
# Check result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single validation check.

    Attributes:
        name:    Short identifier (e.g. ``"model_forward_pass"``).
        passed:  Whether the check succeeded.
        detail:  Human-readable explanation of the result.
        elapsed: Wall-clock time in seconds for this check.
    """

    name: str
    passed: bool
    detail: str
    elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "elapsed": round(self.elapsed, 3),
        }


# ---------------------------------------------------------------------------
# Sandbox execution helper
# ---------------------------------------------------------------------------

async def _run_script_in_sandbox(
    script_code: str,
    project_dir: Path,
    timeout: int | None = None,
) -> Dict[str, Any]:
    """Execute a Python script inside the configured sandbox environment.

    If Docker is available and a sandbox image is configured, the script
    is run inside a container with the project directory mounted at
    ``/workspace``.  Otherwise it falls back to a local subprocess with
    a restricted timeout.

    Args:
        script_code: The full Python source code to execute.
        project_dir: Path to the project directory (will be mounted).
        timeout:     Execution timeout in seconds.

    Returns:
        Dictionary with ``success`` (bool), ``stdout`` (str),
        ``stderr`` (str), and ``returncode`` (int).
    """
    timeout = timeout or settings.sandbox_timeout

    # Write the script to a temporary file inside the project directory
    # so it can be accessed from within the sandbox.
    script_path = project_dir / "_validate_tmp.py"
    script_path.write_text(script_code, encoding="utf-8")

    try:
        # Try Docker sandbox first
        docker_available = await _check_docker()
        if docker_available:
            return await _run_docker(script_path, project_dir, timeout)

        # Fallback: local subprocess with timeout
        logger.warning(
            "Docker not available; falling back to local subprocess execution"
        )
        return await _run_local(script_path, timeout)

    finally:
        # Clean up the temporary script
        if script_path.exists():
            script_path.unlink(missing_ok=True)


async def _check_docker() -> bool:
    """Return True if Docker is available on the host."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=10)
        return proc.returncode == 0
    except Exception:
        return False


async def _run_docker(
    script_path: Path,
    project_dir: Path,
    timeout: int,
) -> Dict[str, Any]:
    """Run a validation script inside a Docker container.

    The project directory is mounted read-only at ``/workspace`` and the
    script is executed with ``python /workspace/<script_name>``.
    """
    container_workspace = "/workspace"
    image = settings.sandbox_image
    memory_limit = settings.sandbox_memory_limit

    cmd = [
        "docker", "run",
        "--rm",
        "--network=none",
        f"--memory={memory_limit}",
        f"--cpus=1",
        "-v", f"{project_dir.resolve()}:{container_workspace}:ro",
        "-w", container_workspace,
        image,
        "python", f"{container_workspace}/{script_path.name}",
    ]

    logger.debug("Docker command: {}", " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return {
            "success": proc.returncode == 0,
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "returncode": proc.returncode or 0,
        }
    except asyncio.TimeoutError:
        logger.warning("Docker execution timed out after {}s", timeout)
        try:
            proc.kill()  # type: ignore[union-attr]
        except Exception:
            pass
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "returncode": -1,
        }
    except Exception as exc:
        logger.error("Docker execution error: {}", exc)
        return {
            "success": False,
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
        }


async def _run_local(script_path: Path, timeout: int) -> Dict[str, Any]:
    """Run a validation script as a local subprocess (fallback)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "python", str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_path.parent),
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return {
            "success": proc.returncode == 0,
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "returncode": proc.returncode or 0,
        }
    except asyncio.TimeoutError:
        logger.warning("Local execution timed out after {}s", timeout)
        try:
            proc.kill()  # type: ignore[union-attr]
        except Exception:
            pass
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "returncode": -1,
        }
    except Exception as exc:
        logger.error("Local execution error: {}", exc)
        return {
            "success": False,
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
        }


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

class ValidationPipeline:
    """Orchestrate end-to-end validation of a generated ML project.

    The pipeline delegates static analysis to the
    :class:`~backend.agents.validation_agent.ValidationAgent` and then
    executes the generated validation scripts inside the sandbox to
    verify runtime correctness.

    Parameters
    ----------
    sandbox_timeout : int | None
        Per-script execution timeout in seconds.  Defaults to
        ``settings.sandbox_timeout``.
    """

    def __init__(self, sandbox_timeout: int | None = None) -> None:
        self._agent = ValidationAgent()
        self._sandbox_timeout = sandbox_timeout or settings.sandbox_timeout
        logger.info(
            "ValidationPipeline initialised (timeout={}s)",
            self._sandbox_timeout,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def validate(
        self,
        source_files: Dict[str, str],
        research_summary: Dict[str, Any] | None = None,
        test_files: Dict[str, str] | None = None,
        project_dir: Path | str | None = None,
    ) -> Dict[str, Any]:
        """Run the full validation pipeline on a generated project.

        This method performs two phases:

        1. **Static validation** via the ``ValidationAgent`` (syntax
           checks, import checks, config integrity, etc.).
        2. **Runtime validation** by executing the agent-generated
           validation scripts (model forward pass, dataset loading,
           training loop init) inside the sandbox.

        Args:
            source_files:     Dict mapping file paths to source code
                              (output of CodingAgent).
            research_summary: Structured paper summary from ResearchAgent
                              (optional, improves script generation).
            test_files:       Test files from TestingAgent (optional).
            project_dir:      Directory to materialise the project into
                              for sandbox execution.  A temporary
                              directory is used if not provided.

        Returns:
            A validation report dictionary with:
            - ``checks_passed`` (int): Number of checks that passed.
            - ``checks_failed`` (int): Number of checks that failed.
            - ``details`` (list[dict]): Per-check result dicts.
            - ``overall_status`` (str): ``"passed"``, ``"failed"``, or
              ``"partial"``.
        """
        logger.info(
            "Starting validation pipeline for {} source files",
            len(source_files),
        )
        start_time = time.monotonic()

        all_checks: List[CheckResult] = []

        # ---- Phase 1: Static validation via agent ----
        logger.info("Phase 1: Running static validation checks")
        static_checks, validation_scripts = await self._run_static_validation(
            source_files=source_files,
            research_summary=research_summary or {},
            test_files=test_files or {},
        )
        all_checks.extend(static_checks)

        # ---- Phase 2: Runtime validation in sandbox ----
        if validation_scripts:
            logger.info(
                "Phase 2: Executing {} validation scripts in sandbox",
                len(validation_scripts),
            )
            runtime_checks = await self._run_runtime_validation(
                source_files=source_files,
                validation_scripts=validation_scripts,
                project_dir=project_dir,
            )
            all_checks.extend(runtime_checks)
        else:
            logger.warning(
                "No validation scripts generated; skipping runtime validation"
            )

        # ---- Aggregate results ----
        report = self._build_report(all_checks, start_time)

        logger.info(
            "Validation complete: {passed}/{total} checks passed -- {status}",
            passed=report["checks_passed"],
            total=report["checks_passed"] + report["checks_failed"],
            status=report["overall_status"],
        )

        return report

    # ------------------------------------------------------------------
    # Phase 1: Static validation
    # ------------------------------------------------------------------

    async def _run_static_validation(
        self,
        source_files: Dict[str, str],
        research_summary: Dict[str, Any],
        test_files: Dict[str, str],
    ) -> tuple[List[CheckResult], Dict[str, str]]:
        """Run the ValidationAgent's static checks.

        Returns:
            A tuple of (list of CheckResult, dict of validation scripts).
        """
        context = {
            "files": source_files,
            "test_files": test_files,
            "research_summary": research_summary,
        }

        t0 = time.monotonic()
        try:
            agent_result = await self._agent.run(context)
        except Exception as exc:
            logger.error("ValidationAgent raised an exception: {}", exc)
            return [
                CheckResult(
                    name="static_validation",
                    passed=False,
                    detail=f"ValidationAgent failed: {exc}",
                    elapsed=time.monotonic() - t0,
                )
            ], {}

        elapsed = time.monotonic() - t0

        # Convert agent check dicts to CheckResult instances
        checks: List[CheckResult] = []
        for check_dict in agent_result.get("checks", []):
            checks.append(
                CheckResult(
                    name=check_dict.get("name", "unknown_check"),
                    passed=check_dict.get("passed", False),
                    detail=check_dict.get("detail", ""),
                    elapsed=elapsed / max(len(agent_result.get("checks", [])), 1),
                )
            )

        validation_scripts: Dict[str, str] = agent_result.get(
            "validation_scripts", {}
        )

        logger.info(
            "Static validation: {}/{} checks passed in {:.2f}s",
            sum(1 for c in checks if c.passed),
            len(checks),
            elapsed,
        )

        return checks, validation_scripts

    # ------------------------------------------------------------------
    # Phase 2: Runtime validation in sandbox
    # ------------------------------------------------------------------

    async def _run_runtime_validation(
        self,
        source_files: Dict[str, str],
        validation_scripts: Dict[str, str],
        project_dir: Path | str | None = None,
    ) -> List[CheckResult]:
        """Execute generated validation scripts in the sandbox.

        Each script is expected to print ``PASS`` or ``FAIL`` as its
        last meaningful output line.

        Args:
            source_files:       The generated project source files.
            validation_scripts: Dict mapping script names to source code.
            project_dir:        Optional pre-existing project directory.

        Returns:
            A list of CheckResult for each executed script.
        """
        checks: List[CheckResult] = []

        # Materialise project files to disk
        use_temp = project_dir is None
        if use_temp:
            tmp = tempfile.mkdtemp(prefix="reagentai_validate_")
            proj_path = Path(tmp)
        else:
            proj_path = Path(project_dir)  # type: ignore[arg-type]

        try:
            # Write all source files to the project directory
            self._materialise_files(proj_path, source_files)

            # Execute each validation script
            for script_name, script_code in validation_scripts.items():
                check = await self._execute_validation_script(
                    script_name=script_name,
                    script_code=script_code,
                    project_dir=proj_path,
                )
                checks.append(check)

        finally:
            # Clean up temp directory if we created it
            if use_temp:
                import shutil
                try:
                    shutil.rmtree(proj_path, ignore_errors=True)
                except Exception:
                    pass

        return checks

    async def _execute_validation_script(
        self,
        script_name: str,
        script_code: str,
        project_dir: Path,
    ) -> CheckResult:
        """Execute a single validation script and interpret the result.

        The script is expected to print ``PASS`` on success or ``FAIL``
        (with an optional error message) on failure.

        Args:
            script_name: Human-readable name for the check.
            script_code: Python source code to execute.
            project_dir: Project root directory.

        Returns:
            A :class:`CheckResult` for this script.
        """
        # Derive a friendly check name from the script filename
        check_name = script_name.replace(".py", "").replace(" ", "_")

        logger.info("Executing validation script: {}", script_name)
        t0 = time.monotonic()

        result = await _run_script_in_sandbox(
            script_code=script_code,
            project_dir=project_dir,
            timeout=self._sandbox_timeout,
        )

        elapsed = time.monotonic() - t0

        stdout = result.get("stdout", "").strip()
        stderr = result.get("stderr", "").strip()
        success = result.get("success", False)

        # Determine pass/fail from script output
        # Scripts are expected to print PASS or FAIL as their last line
        output_lines = stdout.splitlines()
        last_line = output_lines[-1].strip().upper() if output_lines else ""

        if "PASS" in last_line:
            passed = True
            detail = f"Script '{script_name}' executed successfully: PASS"
        elif "FAIL" in last_line:
            passed = False
            # Include the failure message
            fail_detail = stdout if len(stdout) < 500 else stdout[-500:]
            detail = f"Script '{script_name}' reported FAIL: {fail_detail}"
        elif success:
            # Script exited 0 but did not print PASS/FAIL
            passed = True
            detail = (
                f"Script '{script_name}' exited with code 0 "
                f"(no explicit PASS/FAIL found)"
            )
        else:
            passed = False
            error_info = stderr[:500] if stderr else f"Exit code: {result.get('returncode', -1)}"
            detail = f"Script '{script_name}' failed: {error_info}"

        logger.info(
            "Validation script '{}': {} ({:.2f}s)",
            script_name,
            "PASS" if passed else "FAIL",
            elapsed,
        )

        return CheckResult(
            name=check_name,
            passed=passed,
            detail=detail,
            elapsed=elapsed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _materialise_files(
        project_dir: Path,
        files: Dict[str, str],
    ) -> None:
        """Write file contents to disk under *project_dir*.

        Creates intermediate directories as needed.

        Args:
            project_dir: Root directory for the project.
            files:       Dict mapping relative paths to file contents.
        """
        for filepath, content in files.items():
            target = project_dir / filepath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        logger.debug(
            "Materialised {} files under {}",
            len(files),
            project_dir,
        )

    @staticmethod
    def _build_report(
        checks: List[CheckResult],
        start_time: float,
    ) -> Dict[str, Any]:
        """Aggregate individual check results into a validation report.

        Args:
            checks:     All check results from both phases.
            start_time: ``time.monotonic()`` value when validation started.

        Returns:
            A report dictionary with ``checks_passed``, ``checks_failed``,
            ``details``, and ``overall_status``.
        """
        total_elapsed = time.monotonic() - start_time
        passed = [c for c in checks if c.passed]
        failed = [c for c in checks if not c.passed]

        if not checks:
            overall_status = "failed"
        elif len(failed) == 0:
            overall_status = "passed"
        elif len(passed) == 0:
            overall_status = "failed"
        else:
            overall_status = "partial"

        return {
            "checks_passed": len(passed),
            "checks_failed": len(failed),
            "details": [c.to_dict() for c in checks],
            "overall_status": overall_status,
            "total_elapsed": round(total_elapsed, 3),
        }
