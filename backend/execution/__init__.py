"""ReagentAI Execution Package -- Phase 7.

This package provides the sandboxed execution environment for running
generated ML projects.  It uses Docker containers to isolate execution,
enforce resource limits, and capture stdout/stderr output for downstream
analysis by the debug loop.

Modules:
    sandbox     : Docker container lifecycle management and command execution.
    code_runner : High-level runner that installs dependencies, executes the
                  project entry-point, and returns a structured result.
"""

from backend.execution.sandbox import DockerSandbox, SandboxConfig
from backend.execution.code_runner import CodeRunner, ExecutionResult

__all__ = [
    "DockerSandbox",
    "SandboxConfig",
    "CodeRunner",
    "ExecutionResult",
]
