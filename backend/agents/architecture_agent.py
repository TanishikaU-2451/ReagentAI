"""ReagentAI Architecture Agent.

The ArchitectureAgent generates a project directory structure tailored to the
algorithm type identified in the research summary. It uses Mixtral to reason
about which files and directories are needed for a clean, modular PyTorch
project, then returns a structured representation of the directory tree
together with a description of every file's purpose.

Output format::

    {
        "project_name": "paper-impl",
        "directory_tree": "paper-impl/\\n  src/\\n    model.py\\n    ...",
        "files": {
            "src/model.py": "Core model architecture ...",
            "src/trainer.py": "Training loop ...",
            ...
        },
        "algorithm_type": "transformer | cnn | gan | rl | ...",
        "notes": "..."
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from backend.agents.base_agent import BaseAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger


class ArchitectureAgent(BaseAgent):
    """Design the project directory layout for a paper implementation.

    Given the research summary and implementation plan, the agent decides
    which source files, configuration files, and scaffolding are required,
    then outputs a directory tree and file-purpose mapping.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ArchitectureAgent",
            model_id=settings.architecture_agent_model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a project directory structure.

        Args:
            context: Must contain:
                - ``research_summary`` (dict): Output of ResearchAgent.
                Optionally:
                - ``plan`` (dict): Output of PlanningAgent.

        Returns:
            Dictionary with ``project_name``, ``directory_tree``, ``files``,
            ``algorithm_type``, and ``notes``.
        """
        research: Dict[str, Any] = context.get("research_summary", {})
        plan: Dict[str, Any] = context.get("plan", {})

        if not research:
            self.logger.error("No research_summary in context")
            return self._empty_architecture("No research summary provided.")

        self.log_reasoning("start", "Designing project architecture")

        algorithm = research.get("algorithm", "Not specified")
        components = research.get("architecture_components", [])
        datasets = research.get("datasets", [])
        training = research.get("training_strategy", "Not specified")
        plan_steps = plan.get("steps", [])
        estimated_files = plan.get("estimated_files", [])

        components_text = "\n".join(f"  - {c}" for c in components) or "  (none)"
        plan_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan_steps)) or "  (no plan)"
        est_files_text = ", ".join(estimated_files) or "(none)"

        # ----- Build prompt -----
        system_instruction = (
            "You are a senior software architect who specialises in "
            "structuring machine-learning research code into clean, "
            "modular Python/PyTorch projects. Given a research summary "
            "and an implementation plan, design the ideal project "
            "directory layout. Respond ONLY with a valid JSON object."
        )

        user_message = f"""Algorithm: {algorithm}

Architecture Components:
{components_text}

Datasets: {', '.join(d['name'] if isinstance(d, dict) else str(d) for d in datasets) or 'Not specified'}

Training Strategy: {training}

Implementation Plan:
{plan_text}

Estimated files from planner: {est_files_text}

---

Design the project directory structure as a JSON object:

{{
  "project_name": "<short-kebab-case project name>",
  "algorithm_type": "<one of: transformer, cnn, rnn, gan, vae, diffusion, rl, graph_nn, other>",
  "directory_tree": "<ASCII directory tree as a single string with newlines>",
  "files": {{
    "src/model.py": "Description of what this file contains and why",
    "src/layers/<module>.py": "Description ...",
    "src/trainer.py": "Description ...",
    "src/dataset_loader.py": "Description ...",
    "src/utils.py": "Description ...",
    "config.yaml": "Description ...",
    "requirements.txt": "Description ...",
    "README.md": "Description ...",
    "Dockerfile": "Description ...",
    "tests/test_model.py": "Description ...",
    ...
  }},
  "notes": "<Any architecture-level notes or design decisions>"
}}

Rules:
- Include a src/ directory for all Python source code.
- Include a tests/ directory for test files.
- If the model has multiple distinct sub-modules (e.g. encoder, decoder,
  attention), put each in its own file under src/layers/ or src/modules/.
- Always include: model.py, trainer.py, dataset_loader.py, config.yaml,
  requirements.txt, README.md, Dockerfile.
- Keep it practical -- no more than 15 source files.

Return ONLY the JSON object."""

        prompt = self._build_prompt(system_instruction, user_message)

        # ----- Call model -----
        self.log_reasoning("model_call", "Requesting architecture design from Mixtral")
        raw_output = await self.call_model(prompt, max_new_tokens=2048, temperature=0.25)
        self.log_reasoning("model_response", f"Received {len(raw_output)} chars")

        # ----- Parse response -----
        result = self._parse_architecture(raw_output)
        result["raw_model_output"] = raw_output

        self.log_reasoning(
            "complete",
            f"Architecture: {result.get('project_name', '?')} "
            f"({len(result.get('files', {}))} files)",
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_architecture(self, raw: str) -> Dict[str, Any]:
        """Parse the model output into an architecture dict."""
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

        self.logger.warning("Could not parse architecture JSON; returning fallback")
        return self._empty_architecture("Failed to parse architecture output.")

    @staticmethod
    def _empty_architecture(reason: str = "") -> Dict[str, Any]:
        """Return a minimal fallback architecture."""
        return {
            "project_name": "paper-impl",
            "algorithm_type": "other",
            "directory_tree": (
                "paper-impl/\n"
                "  src/\n"
                "    model.py\n"
                "    trainer.py\n"
                "    dataset_loader.py\n"
                "    utils.py\n"
                "  tests/\n"
                "    test_model.py\n"
                "  config.yaml\n"
                "  requirements.txt\n"
                "  README.md\n"
                "  Dockerfile\n"
            ),
            "files": {
                "src/model.py": "Core model architecture",
                "src/trainer.py": "Training loop",
                "src/dataset_loader.py": "Dataset loading and preprocessing",
                "src/utils.py": "Utility helpers",
                "tests/test_model.py": "Model unit tests",
                "config.yaml": "Configuration file",
                "requirements.txt": "Python dependencies",
                "README.md": "Project documentation",
                "Dockerfile": "Container definition",
            },
            "notes": reason,
            "raw_model_output": "",
        }
