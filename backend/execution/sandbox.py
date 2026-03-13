"""ReagentAI Docker Sandbox -- Phase 7.

Manages Docker containers for safely executing generated ML projects.

Features:

* Creates containers from the ``python:3.11-slim`` base image (or whatever
  image is configured in ``settings.sandbox_image``).
* Bind-mounts the generated project directory into the container at
  ``/workspace``.
* Enforces resource limits (memory, CPU count, execution timeout) derived
  from ``settings``.
* Provides a high-level :meth:`run_in_sandbox` method that executes an
  arbitrary shell command inside the container and returns stdout / stderr.
* Automatically removes containers after execution to avoid resource leaks.

Usage::

    from backend.execution.sandbox import DockerSandbox

    sandbox = DockerSandbox()
    result = await sandbox.run_in_sandbox(
        project_path="/abs/path/to/project",
        command="python src/trainer.py",
    )
    print(result.stdout)
    print(result.stderr)
    print(result.exit_code)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import docker
from docker.errors import (
    APIError,
    ContainerError,
    DockerException,
    ImageNotFound,
    NotFound,
)
from docker.models.containers import Container

from backend.config.settings import settings
from backend.utils.logging import get_logger

logger = get_logger("execution.sandbox")


# ---------------------------------------------------------------------------
# Configuration data class
# ---------------------------------------------------------------------------

@dataclass
class SandboxConfig:
    """Configuration for the Docker sandbox environment.

    Attributes:
        image:         Docker image name/tag.
        memory_limit:  Memory limit string (e.g. ``"2g"``, ``"512m"``).
        cpu_count:     Number of CPUs to allocate.
        timeout:       Maximum execution time in seconds.
        network_mode:  Docker network mode (``"none"`` disables networking).
        work_dir:      Working directory inside the container.
    """

    image: str = ""
    memory_limit: str = ""
    cpu_count: int = 2
    timeout: int = 0
    network_mode: str = "none"
    work_dir: str = "/workspace"

    def __post_init__(self) -> None:
        """Fill defaults from application settings if not supplied."""
        if not self.image:
            self.image = settings.sandbox_image
        if not self.memory_limit:
            self.memory_limit = settings.sandbox_memory_limit
        if self.timeout <= 0:
            self.timeout = settings.sandbox_timeout


@dataclass
class SandboxResult:
    """Result of a single command execution inside the sandbox.

    Attributes:
        exit_code:       Process exit code (``0`` = success).
        stdout:          Captured standard output.
        stderr:          Captured standard error.
        timed_out:       ``True`` if the command exceeded the timeout.
        execution_time:  Wall-clock seconds the command took.
        container_id:    Docker container ID (for diagnostics).
    """

    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    execution_time: float = 0.0
    container_id: str = ""


# ---------------------------------------------------------------------------
# Docker sandbox
# ---------------------------------------------------------------------------

class DockerSandbox:
    """Manage Docker containers for sandboxed code execution.

    A new container is created for every :meth:`run_in_sandbox` invocation
    and is removed after the command finishes (or times out).  This ensures
    a clean environment for each execution attempt and avoids stale state
    leaking between debug-loop iterations.
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        """Initialise the sandbox manager.

        Args:
            config: Optional sandbox configuration.  If ``None``, settings
                    are read from the application ``Settings`` object.
        """
        self._config = config or SandboxConfig()
        self._client: Optional[docker.DockerClient] = None
        logger.info(
            "DockerSandbox initialised (image={}, mem={}, cpu={}, timeout={}s)",
            self._config.image,
            self._config.memory_limit,
            self._config.cpu_count,
            self._config.timeout,
        )

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> docker.DockerClient:
        """Lazily create and return a Docker client.

        Returns:
            A connected :class:`docker.DockerClient`.

        Raises:
            DockerException: If the Docker daemon is not reachable.
        """
        if self._client is None:
            try:
                self._client = docker.from_env()
                self._client.ping()
                logger.info("Docker daemon connection established")
            except DockerException as exc:
                logger.error("Cannot connect to Docker daemon: {}", exc)
                raise
        return self._client

    def _ensure_image(self) -> None:
        """Pull the sandbox image if it is not already available locally.

        Raises:
            ImageNotFound: If the image cannot be found locally or remotely.
            APIError:      On Docker API failures.
        """
        client = self._get_client()
        image_name = self._config.image

        try:
            client.images.get(image_name)
            logger.debug("Image '{}' found locally", image_name)
        except ImageNotFound:
            logger.info("Pulling image '{}' ...", image_name)
            try:
                client.images.pull(image_name)
                logger.info("Image '{}' pulled successfully", image_name)
            except APIError as exc:
                logger.error("Failed to pull image '{}': {}", image_name, exc)
                raise

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def _create_container(
        self,
        project_path: str,
        command: str,
    ) -> Container:
        """Create a Docker container configured for sandboxed execution.

        The generated project directory is bind-mounted read-write at the
        container's ``/workspace`` path.

        Args:
            project_path: Absolute host path to the generated project.
            command:       Shell command to execute inside the container.

        Returns:
            A created (but not yet started) :class:`Container`.
        """
        client = self._get_client()
        self._ensure_image()

        host_path = str(Path(project_path).resolve())

        container = client.containers.create(
            image=self._config.image,
            command=["bash", "-c", command],
            working_dir=self._config.work_dir,
            volumes={
                host_path: {
                    "bind": self._config.work_dir,
                    "mode": "rw",
                }
            },
            mem_limit=self._config.memory_limit,
            nano_cpus=int(self._config.cpu_count * 1e9),
            network_mode=self._config.network_mode,
            detach=True,
            stdout=True,
            stderr=True,
        )

        logger.info(
            "Container created: id={} image={} mount={}:{}",
            container.short_id,
            self._config.image,
            host_path,
            self._config.work_dir,
        )
        return container

    @staticmethod
    def _remove_container(container: Container) -> None:
        """Force-remove a container, suppressing errors.

        Args:
            container: The Docker container to remove.
        """
        try:
            container.remove(force=True)
            logger.debug("Container {} removed", container.short_id)
        except NotFound:
            logger.debug(
                "Container {} already removed", container.short_id
            )
        except APIError as exc:
            logger.warning(
                "Failed to remove container {}: {}",
                container.short_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run_in_sandbox(
        self,
        project_path: str,
        command: str,
        *,
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """Execute a command inside an isolated Docker container.

        Creates a fresh container, starts it, waits for completion (or
        timeout), captures stdout/stderr, and removes the container.

        Args:
            project_path: Absolute path to the generated project directory
                          on the host filesystem.
            command:       Shell command to execute (e.g.
                          ``"python src/trainer.py"``).
            timeout:       Override the configured timeout (seconds).

        Returns:
            A :class:`SandboxResult` with stdout, stderr, exit code, and
            timing information.
        """
        effective_timeout = timeout if timeout is not None else self._config.timeout
        start_time = time.monotonic()

        logger.info(
            "Running in sandbox: '{}' (timeout={}s, project={})",
            command,
            effective_timeout,
            project_path,
        )

        loop = asyncio.get_running_loop()
        container: Optional[Container] = None

        try:
            # Create and start the container (blocking Docker SDK calls
            # are run in the default thread-pool executor).
            container = await loop.run_in_executor(
                None,
                lambda: self._create_container(project_path, command),
            )

            await loop.run_in_executor(None, container.start)
            logger.debug("Container {} started", container.short_id)

            # Wait for the container to finish, with timeout
            try:
                exit_info = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: container.wait(timeout=effective_timeout),
                    ),
                    timeout=effective_timeout + 5,  # small grace period
                )
                exit_code = exit_info.get("StatusCode", -1)
                timed_out = False
            except (asyncio.TimeoutError, Exception) as wait_exc:
                logger.warning(
                    "Container {} timed out or errored during wait: {}",
                    container.short_id,
                    wait_exc,
                )
                # Attempt to stop the container
                try:
                    await loop.run_in_executor(
                        None, lambda: container.stop(timeout=5)
                    )
                except Exception:
                    pass
                exit_code = -1
                timed_out = True

            # Capture logs
            stdout = await loop.run_in_executor(
                None,
                lambda: container.logs(stdout=True, stderr=False).decode(
                    "utf-8", errors="replace"
                ),
            )
            stderr = await loop.run_in_executor(
                None,
                lambda: container.logs(stdout=False, stderr=True).decode(
                    "utf-8", errors="replace"
                ),
            )

            elapsed = time.monotonic() - start_time

            result = SandboxResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                execution_time=elapsed,
                container_id=container.id,
            )

            log_level = "info" if exit_code == 0 else "warning"
            getattr(logger, log_level)(
                "Sandbox execution finished: exit_code={}, timed_out={}, "
                "elapsed={:.1f}s, stdout_len={}, stderr_len={}",
                exit_code,
                timed_out,
                elapsed,
                len(stdout),
                len(stderr),
            )

            return result

        except DockerException as exc:
            elapsed = time.monotonic() - start_time
            logger.error("Docker error during sandbox execution: {}", exc)
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Docker error: {exc}",
                timed_out=False,
                execution_time=elapsed,
                container_id=container.id if container else "",
            )

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.error("Unexpected error during sandbox execution: {}", exc)
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Unexpected error: {exc}",
                timed_out=False,
                execution_time=elapsed,
                container_id=container.id if container else "",
            )

        finally:
            # Always clean up the container
            if container is not None:
                await loop.run_in_executor(
                    None, lambda: self._remove_container(container)
                )

    # ------------------------------------------------------------------
    # Bulk cleanup
    # ------------------------------------------------------------------

    async def cleanup_all(self, label_prefix: str = "reagentai") -> int:
        """Remove all containers matching a label prefix.

        This is a housekeeping utility to clean up orphaned containers
        from previous runs.

        Args:
            label_prefix: Containers whose name starts with this prefix
                          will be removed.

        Returns:
            The number of containers removed.
        """
        loop = asyncio.get_running_loop()
        client = self._get_client()

        def _do_cleanup() -> int:
            removed = 0
            for container in client.containers.list(all=True):
                if container.name and container.name.startswith(label_prefix):
                    self._remove_container(container)
                    removed += 1
            return removed

        removed = await loop.run_in_executor(None, _do_cleanup)
        logger.info("Cleaned up {} orphaned containers", removed)
        return removed

    def close(self) -> None:
        """Close the Docker client connection."""
        if self._client is not None:
            try:
                self._client.close()
                logger.debug("Docker client closed")
            except Exception:
                pass
            self._client = None
