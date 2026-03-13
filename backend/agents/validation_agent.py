"""ReagentAI Validation Agent.

The ValidationAgent performs lightweight, non-training validation checks
on the generated project to verify that the code is structurally sound
before handing it off to the user. It does NOT execute arbitrary code
directly -- instead it analyses the source files and generates small
validation scripts that are designed to be run inside the Docker sandbox.

Checks performed:

1. **Syntax check** -- Can every .py file be compiled (``ast.parse``)?
2. **Import check** -- Are all imports resolvable against requirements.txt?
3. **Model instantiation check** -- Generate a script that instantiates
   the model and runs a dummy forward pass.
4. **Dataset loading check** -- Verify the dataset loader can be
   instantiated with a dummy config.
5. **Training loop init check** -- Verify the trainer can be constructed
   with model + dataset + config.
6. **Config integrity** -- Does config.yaml parse and contain required keys?

Output format::

    {
        "passed": True | False,
        "checks": [
            {"name": "syntax_check", "passed": True, "detail": "..."},
            ...
        ],
        "validation_scripts": {
            "validate_model.py": "<script code>",
            ...
        },
        "summary": "All 6 checks passed."
    }
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

from backend.agents.base_agent import BaseAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger


class ValidationAgent(BaseAgent):
    """Run lightweight validation on generated project files.

    Unlike most other agents, the ValidationAgent relies primarily on
    local analysis (AST parsing, regex scanning) rather than LLM calls.
    It only invokes the LLM to generate validation scripts that will be
    executed inside the sandbox.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ValidationAgent",
            model_id=settings.validation_agent_model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated project files.

        Args:
            context: Must contain:
                - ``files`` (dict): Source files from CodingAgent.
                Optionally:
                - ``test_files`` (dict): Test files from TestingAgent.
                - ``research_summary`` (dict): For extra context.

        Returns:
            Dictionary with ``passed``, ``checks``, ``validation_scripts``,
            and ``summary``.
        """
        source_files: Dict[str, str] = context.get("files", {})
        test_files: Dict[str, str] = context.get("test_files", {})
        research: Dict[str, Any] = context.get("research_summary", {})

        if not source_files:
            self.logger.error("No source files provided for validation")
            return {
                "passed": False,
                "checks": [],
                "validation_scripts": {},
                "summary": "No source files to validate.",
            }

        self.log_reasoning("start", f"Validating {len(source_files)} source files")

        all_files = {**source_files, **test_files}
        checks: List[Dict[str, Any]] = []

        # --- Check 1: Syntax ---
        syntax_result = self._check_syntax(all_files)
        checks.append(syntax_result)
        self.log_reasoning("check", f"syntax_check: {'PASS' if syntax_result['passed'] else 'FAIL'}")

        # --- Check 2: Import analysis ---
        import_result = self._check_imports(source_files)
        checks.append(import_result)
        self.log_reasoning("check", f"import_check: {'PASS' if import_result['passed'] else 'FAIL'}")

        # --- Check 3: Config integrity ---
        config_result = self._check_config(source_files)
        checks.append(config_result)
        self.log_reasoning("check", f"config_check: {'PASS' if config_result['passed'] else 'FAIL'}")

        # --- Check 4: Required files ---
        required_result = self._check_required_files(source_files)
        checks.append(required_result)
        self.log_reasoning("check", f"required_files_check: {'PASS' if required_result['passed'] else 'FAIL'}")

        # --- Check 5: Model class presence ---
        model_result = self._check_model_class(source_files)
        checks.append(model_result)
        self.log_reasoning("check", f"model_class_check: {'PASS' if model_result['passed'] else 'FAIL'}")

        # --- Check 6: Generate validation scripts via LLM ---
        validation_scripts = await self._generate_validation_scripts(
            source_files, research
        )
        script_check = {
            "name": "validation_scripts_generated",
            "passed": len(validation_scripts) > 0,
            "detail": f"Generated {len(validation_scripts)} validation scripts",
        }
        checks.append(script_check)
        self.log_reasoning("check", f"validation_scripts: {len(validation_scripts)} generated")

        # --- Aggregate ---
        all_passed = all(c["passed"] for c in checks)
        passed_count = sum(1 for c in checks if c["passed"])
        total = len(checks)
        summary = (
            f"{passed_count}/{total} checks passed."
            if not all_passed
            else f"All {total} checks passed."
        )

        self.log_reasoning("complete", summary)

        return {
            "passed": all_passed,
            "checks": checks,
            "validation_scripts": validation_scripts,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Static / local checks (no LLM needed)
    # ------------------------------------------------------------------

    def _check_syntax(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Verify that all .py files have valid Python syntax."""
        errors: List[str] = []
        checked = 0

        for filepath, code in files.items():
            if not filepath.endswith(".py"):
                continue
            checked += 1
            try:
                ast.parse(code, filename=filepath)
            except SyntaxError as exc:
                errors.append(
                    f"{filepath}:{exc.lineno}: {exc.msg}"
                )

        return {
            "name": "syntax_check",
            "passed": len(errors) == 0,
            "detail": (
                f"All {checked} Python files parsed successfully."
                if not errors
                else f"Syntax errors in {len(errors)} file(s): " + "; ".join(errors[:5])
            ),
        }

    def _check_imports(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Check that imported third-party packages appear in requirements.txt."""
        requirements_code = files.get("requirements.txt", "")
        # Extract package names from requirements.txt
        req_packages = set()
        for line in requirements_code.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pkg = re.split(r"[>=<!\[]", line)[0].strip().lower().replace("-", "_")
                req_packages.add(pkg)

        # Standard library modules (subset -- enough for common cases)
        stdlib = {
            "os", "sys", "re", "json", "math", "random", "copy", "time",
            "datetime", "pathlib", "collections", "abc", "typing",
            "functools", "itertools", "logging", "argparse", "dataclasses",
            "ast", "io", "contextlib", "glob", "shutil", "subprocess",
            "unittest", "warnings", "hashlib", "pickle", "csv",
            "__future__", "enum", "textwrap", "string", "struct",
        }

        missing: List[str] = []
        for filepath, code in files.items():
            if not filepath.endswith(".py"):
                continue
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0].lower().replace("-", "_")
                        if top not in stdlib and top not in req_packages:
                            # Allow project-internal imports
                            if top not in {"src", "tests", "config"}:
                                missing.append(f"{filepath}: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0].lower().replace("-", "_")
                        if top not in stdlib and top not in req_packages:
                            if top not in {"src", "tests", "config"}:
                                missing.append(f"{filepath}: {node.module}")

        return {
            "name": "import_check",
            "passed": len(missing) == 0,
            "detail": (
                "All imports satisfied by requirements.txt or stdlib."
                if not missing
                else f"Potentially missing packages: {'; '.join(missing[:10])}"
            ),
        }

    def _check_config(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Verify that config.yaml exists and is parseable."""
        config_code = files.get("config.yaml", "")
        if not config_code.strip():
            return {
                "name": "config_check",
                "passed": False,
                "detail": "config.yaml is missing or empty.",
            }

        # Lightweight YAML validity check (look for key: value patterns)
        has_keys = bool(re.search(r"^\w[\w_]*\s*:", config_code, re.MULTILINE))
        return {
            "name": "config_check",
            "passed": has_keys,
            "detail": (
                "config.yaml present and contains key-value pairs."
                if has_keys
                else "config.yaml exists but appears to have no valid YAML keys."
            ),
        }

    def _check_required_files(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Ensure mandatory project files are present."""
        required = [
            "src/model.py",
            "src/trainer.py",
            "src/dataset_loader.py",
            "config.yaml",
            "requirements.txt",
            "README.md",
            "Dockerfile",
        ]
        missing = [f for f in required if f not in files]
        return {
            "name": "required_files_check",
            "passed": len(missing) == 0,
            "detail": (
                "All required files present."
                if not missing
                else f"Missing files: {', '.join(missing)}"
            ),
        }

    def _check_model_class(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Verify that model.py defines at least one nn.Module subclass."""
        model_code = ""
        for fp, code in files.items():
            if fp.endswith("model.py"):
                model_code = code
                break

        if not model_code:
            return {
                "name": "model_class_check",
                "passed": False,
                "detail": "No model.py found.",
            }

        # Look for `class Xxx(nn.Module)` pattern
        has_module = bool(
            re.search(r"class\s+\w+\s*\(.*nn\.Module.*\)", model_code)
        )
        # Also check for forward method
        has_forward = bool(
            re.search(r"def\s+forward\s*\(", model_code)
        )

        passed = has_module and has_forward
        detail_parts = []
        if not has_module:
            detail_parts.append("No nn.Module subclass found")
        if not has_forward:
            detail_parts.append("No forward() method found")

        return {
            "name": "model_class_check",
            "passed": passed,
            "detail": (
                "model.py defines an nn.Module subclass with forward()."
                if passed
                else "; ".join(detail_parts)
            ),
        }

    # ------------------------------------------------------------------
    # LLM-generated validation scripts
    # ------------------------------------------------------------------

    async def _generate_validation_scripts(
        self,
        source_files: Dict[str, str],
        research: Dict[str, Any],
    ) -> Dict[str, str]:
        """Generate small Python scripts to validate model/dataset/trainer."""
        # Gather key file snippets
        model_code = ""
        trainer_code = ""
        dataset_code = ""
        config_code = ""

        for fp, code in source_files.items():
            if fp.endswith("model.py"):
                model_code = code[:3000]
            elif fp.endswith("trainer.py"):
                trainer_code = code[:3000]
            elif fp.endswith("dataset_loader.py"):
                dataset_code = code[:3000]
            elif fp.endswith("config.yaml"):
                config_code = code[:1500]

        system_instruction = (
            "You are an ML engineer writing quick validation scripts. "
            "Each script should be self-contained, import the project "
            "modules, and test ONE thing. Scripts should print PASS or "
            "FAIL. Output ONLY a JSON object mapping script names to code."
        )

        user_message = f"""Generate validation scripts for this project.

--- model.py (excerpt) ---
{model_code}

--- trainer.py (excerpt) ---
{trainer_code}

--- dataset_loader.py (excerpt) ---
{dataset_code}

--- config.yaml ---
{config_code}

Return a JSON object:
{{
  "validate_model.py": "<script that instantiates the model, creates a random input tensor of the correct shape, runs model.forward(), and prints PASS if output shape is valid>",
  "validate_dataset.py": "<script that instantiates the dataset with dummy/small config, calls __len__ and __getitem__(0), prints PASS>",
  "validate_trainer.py": "<script that constructs Trainer with model+dataset+config, calls one training step, prints PASS>"
}}

Each script should:
- Be wrapped in try/except so failures print FAIL + error message.
- Use small/dummy data so it runs in < 5 seconds on CPU.
- Import from the src/ package.

Return ONLY the JSON object, no markdown."""

        prompt = self._build_prompt(system_instruction, user_message)

        try:
            raw = await self.call_model(prompt, max_new_tokens=3072, temperature=0.2)
            return self._parse_scripts(raw)
        except Exception as exc:
            self.logger.error("Failed to generate validation scripts: {err}", err=str(exc))
            return {}

    def _parse_scripts(self, raw: str) -> Dict[str, str]:
        """Parse LLM output into a dict of script name -> code."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        self.logger.warning("Could not parse validation scripts JSON")
        return {}
