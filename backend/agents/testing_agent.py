"""ReagentAI Testing Agent.

The TestingAgent generates a comprehensive test suite for the implementation
produced by the CodingAgent. It inspects the generated source files and
creates pytest-compatible test cases covering:

* Model instantiation and forward pass (shape checks)
* Dataset loading and batch iteration
* Trainer initialisation and single-step sanity checks
* Configuration loading
* Edge cases and error handling

Output format::

    {
        "test_files": {
            "tests/test_model.py": "<test code>",
            "tests/test_dataset.py": "<test code>",
            "tests/test_trainer.py": "<test code>",
            "tests/conftest.py": "<shared fixtures>",
        },
        "generation_log": [...]
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from backend.agents.base_agent import BaseAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger


class TestingAgent(BaseAgent):
    """Generate pytest test suites for a generated ML project.

    The agent reads the generated source files, identifies testable
    classes and functions, and produces test modules that exercise the
    public API of each source file.
    """

    def __init__(self) -> None:
        super().__init__(
            name="TestingAgent",
            model_id=settings.testing_agent_model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test files for the project.

        Args:
            context: Must contain:
                - ``files`` (dict): Mapping of filepath -> code from CodingAgent.
                Optionally:
                - ``research_summary`` (dict): Research context.
                - ``architecture`` (dict): Architecture context.

        Returns:
            Dictionary with ``test_files`` and ``generation_log``.
        """
        source_files: Dict[str, str] = context.get("files", {})
        research: Dict[str, Any] = context.get("research_summary", {})
        architecture: Dict[str, Any] = context.get("architecture", {})

        if not source_files:
            self.logger.error("No source files provided in context")
            return {"test_files": {}, "generation_log": [], "error": "No source files"}

        self.log_reasoning("start", f"Generating tests for {len(source_files)} source files")

        # Identify which source files need tests
        testable_files = {
            fp: code
            for fp, code in source_files.items()
            if fp.endswith(".py") and not fp.startswith("tests/")
        }

        test_files: Dict[str, str] = {}
        generation_log: List[Dict[str, Any]] = []

        # Generate conftest.py first with shared fixtures
        self.log_reasoning("generate_file", "Generating tests/conftest.py")
        conftest_code = await self._generate_conftest(testable_files, research)
        test_files["tests/conftest.py"] = conftest_code
        generation_log.append(
            {"file": "tests/conftest.py", "status": "ok", "chars": len(conftest_code)}
        )

        # Generate test file for each source file
        for filepath, source_code in testable_files.items():
            test_filepath = self._source_to_test_path(filepath)
            self.log_reasoning("generate_file", f"Generating {test_filepath}")

            try:
                test_code = await self._generate_test_file(
                    source_filepath=filepath,
                    source_code=source_code,
                    all_source_files=testable_files,
                    research=research,
                )
                test_files[test_filepath] = test_code
                generation_log.append(
                    {"file": test_filepath, "status": "ok", "chars": len(test_code)}
                )
                self.log_reasoning("file_done", f"{test_filepath}: {len(test_code)} chars")
            except Exception as exc:
                self.logger.error(
                    "Failed to generate {file}: {err}",
                    file=test_filepath,
                    err=str(exc),
                )
                generation_log.append(
                    {"file": test_filepath, "status": "error", "error": str(exc)}
                )

        self.log_reasoning(
            "complete",
            f"Generated {len(test_files)} test files",
        )

        return {
            "test_files": test_files,
            "generation_log": generation_log,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_conftest(
        self,
        source_files: Dict[str, str],
        research: Dict[str, Any],
    ) -> str:
        """Generate a conftest.py with shared pytest fixtures."""
        # Collect class/function names from source files for fixture targets
        file_summaries = []
        for fp, code in source_files.items():
            first_lines = "\n".join(code.splitlines()[:40])
            file_summaries.append(f"--- {fp} ---\n{first_lines}")
        summaries_text = "\n\n".join(file_summaries)[:5000]

        hyperparams = research.get("hyperparameters", {})

        system_instruction = (
            "You are an expert Python test engineer. Generate a conftest.py "
            "file with shared pytest fixtures for an ML project. "
            "Output ONLY the Python code -- no markdown fences."
        )

        user_message = f"""Source files overview:
{summaries_text}

Hyperparameters: {json.dumps(hyperparams, indent=2, default=str)}

Generate tests/conftest.py with fixtures for:
1. A sample config dict matching the project's config.yaml structure.
2. A small random input tensor for model forward-pass testing.
3. A mock dataset with a few synthetic samples.
4. A device fixture (cpu for CI, cuda if available).
5. Any other reusable fixtures.

Use @pytest.fixture decorator. Include docstrings.
Output ONLY the raw Python code."""

        prompt = self._build_prompt(system_instruction, user_message)
        raw = await self.call_model(prompt, max_new_tokens=2048, temperature=0.2)
        return self._strip_fences(raw)

    async def _generate_test_file(
        self,
        source_filepath: str,
        source_code: str,
        all_source_files: Dict[str, str],
        research: Dict[str, Any],
    ) -> str:
        """Generate a test file for a specific source file."""
        system_instruction = (
            "You are an expert Python test engineer. Generate comprehensive "
            "pytest test cases for the given source file. "
            "Output ONLY the Python code -- no markdown fences."
        )

        # Trim source code if very long
        trimmed_source = source_code[:6000]

        user_message = f"""SOURCE FILE: {source_filepath}
```
{trimmed_source}
```

Generate a pytest test file with:
1. Tests for every public class and function in the source file.
2. Shape checks for tensor-returning functions (assert output.shape == expected).
3. Type checks where appropriate.
4. Edge cases: empty input, single sample, batch of 1.
5. A test that the model/class can be instantiated with default config.
6. If it is a model file: test forward pass with random input, test output shape,
   test gradient flow (loss.backward() does not raise).
7. If it is a trainer file: test that a single training step runs without error.
8. If it is a dataset file: test __len__ and __getitem__.

Use pytest parametrize where applicable.
Use fixtures from conftest.py.
All tests should be fast (use small tensors, few iterations).
Output ONLY the raw Python code."""

        prompt = self._build_prompt(system_instruction, user_message)
        raw = await self.call_model(prompt, max_new_tokens=3072, temperature=0.2)
        return self._strip_fences(raw)

    @staticmethod
    def _source_to_test_path(source_path: str) -> str:
        """Convert a source file path to a test file path.

        ``src/model.py`` -> ``tests/test_model.py``
        ``src/layers/attention.py`` -> ``tests/test_attention.py``
        """
        filename = source_path.rsplit("/", 1)[-1]
        if not filename.startswith("test_"):
            filename = f"test_{filename}"
        return f"tests/{filename}"

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences."""
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()
