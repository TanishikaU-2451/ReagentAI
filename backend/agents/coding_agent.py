"""ReagentAI Coding Agent.

The CodingAgent is the core code-generation component of the pipeline.
It receives the research summary, implementation plan, and architecture
layout, then uses CodeLlama to generate every source file listed in the
architecture specification.

Because individual files can be long, the agent iterates over the file
list, generating one file at a time via the LLM, and accumulates the
results into a single dictionary mapping relative file paths to their
full source-code content.

Output format::

    {
        "files": {
            "src/model.py": "import torch\\n...",
            "src/trainer.py": "...",
            "config.yaml": "...",
            "requirements.txt": "...",
            "README.md": "...",
            "Dockerfile": "...",
            ...
        },
        "generation_log": [
            {"file": "src/model.py", "status": "ok", "tokens": 1234},
            ...
        ]
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from backend.agents.base_agent import BaseAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger


# Files that must always be generated regardless of architecture output
_MANDATORY_FILES = [
    "src/model.py",
    "src/trainer.py",
    "src/dataset_loader.py",
    "config.yaml",
    "requirements.txt",
    "README.md",
    "Dockerfile",
]


class CodingAgent(BaseAgent):
    """Generate complete source code for every file in the project.

    The agent processes files one at a time so that each generation call
    stays within the model's context window. Earlier generated files are
    summarised and fed forward as context for later files to maintain
    coherence across the codebase.
    """

    def __init__(self) -> None:
        super().__init__(
            name="CodingAgent",
            model_id=settings.coding_agent_model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate implementation files for the entire project.

        Args:
            context: Must contain:
                - ``research_summary`` (dict): Output of ResearchAgent.
                - ``plan`` (dict): Output of PlanningAgent.
                - ``architecture`` (dict): Output of ArchitectureAgent.

        Returns:
            Dictionary with ``files`` (path -> code) and ``generation_log``.
        """
        research: Dict[str, Any] = context.get("research_summary", {})
        plan: Dict[str, Any] = context.get("plan", {})
        architecture: Dict[str, Any] = context.get("architecture", {})

        if not research:
            self.logger.error("No research_summary in context")
            return {"files": {}, "generation_log": [], "error": "Missing research summary"}

        self.log_reasoning("start", "Beginning code generation")

        # Determine the file list from architecture, ensuring mandatory files
        arch_files: Dict[str, str] = architecture.get("files", {})
        file_list = list(arch_files.keys())
        for mf in _MANDATORY_FILES:
            if mf not in file_list:
                file_list.append(mf)

        # Build a compact context string shared across all file generations
        shared_context = self._build_shared_context(research, plan, architecture)

        generated_files: Dict[str, str] = {}
        generation_log: List[Dict[str, Any]] = []

        for filepath in file_list:
            file_desc = arch_files.get(filepath, f"Implementation file: {filepath}")
            self.log_reasoning("generate_file", f"Generating {filepath}")

            try:
                code = await self._generate_file(
                    filepath=filepath,
                    file_description=file_desc,
                    shared_context=shared_context,
                    already_generated=generated_files,
                )
                generated_files[filepath] = code
                generation_log.append(
                    {"file": filepath, "status": "ok", "chars": len(code)}
                )
                self.log_reasoning(
                    "file_done",
                    f"{filepath}: {len(code)} chars generated",
                )
            except Exception as exc:
                self.logger.error(
                    "Failed to generate {file}: {err}",
                    file=filepath,
                    err=str(exc),
                )
                generation_log.append(
                    {"file": filepath, "status": "error", "error": str(exc)}
                )

        self.log_reasoning(
            "complete",
            f"Generated {len(generated_files)}/{len(file_list)} files",
        )

        return {
            "files": generated_files,
            "generation_log": generation_log,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_shared_context(
        self,
        research: Dict[str, Any],
        plan: Dict[str, Any],
        architecture: Dict[str, Any],
    ) -> str:
        """Build a condensed context string for the coding prompt."""
        algorithm = research.get("algorithm", "Not specified")
        components = research.get("architecture_components", [])
        hyperparams = research.get("hyperparameters", {})
        training = research.get("training_strategy", "Not specified")
        equations = research.get("equations", [])

        steps = plan.get("steps", [])
        project_name = architecture.get("project_name", "paper-impl")
        algo_type = architecture.get("algorithm_type", "other")

        equations_block = ""
        for eq in equations:
            if isinstance(eq, dict):
                equations_block += (
                    f"  - {eq.get('name', 'eq')}: {eq.get('latex', '')} "
                    f"({eq.get('description', '')})\n"
                )
            else:
                equations_block += f"  - {eq}\n"

        return f"""PROJECT: {project_name}
ALGORITHM TYPE: {algo_type}
ALGORITHM: {algorithm}

ARCHITECTURE COMPONENTS:
{chr(10).join('  - ' + c for c in components)}

HYPERPARAMETERS:
{json.dumps(hyperparams, indent=2, default=str)}

TRAINING STRATEGY: {training}

KEY EQUATIONS:
{equations_block}
IMPLEMENTATION STEPS:
{chr(10).join('  ' + s for s in steps[:15])}"""

    async def _generate_file(
        self,
        filepath: str,
        file_description: str,
        shared_context: str,
        already_generated: Dict[str, str],
    ) -> str:
        """Generate a single project file.

        Args:
            filepath:          Relative path of the file to generate.
            file_description:  One-line description from the architecture agent.
            shared_context:    Condensed research/plan/architecture context.
            already_generated: Dict of files already generated (for coherence).

        Returns:
            The generated file content as a string.
        """
        # Provide summaries of already-generated files for cross-file coherence
        prev_summary = ""
        if already_generated:
            summaries = []
            for fp, code in already_generated.items():
                # Include first 30 lines as a summary
                snippet = "\n".join(code.splitlines()[:30])
                summaries.append(f"--- {fp} (first 30 lines) ---\n{snippet}")
            # Limit total summary size
            prev_summary = "\n\n".join(summaries)[:4000]

        system_instruction = (
            "You are an ML engineer. Generate complete, runnable Python code. "
            "Output ONLY the code - no markdown, no explanations."
        )

        user_message = f"""{shared_context}

FILE: {filepath}
PURPOSE: {file_description}

"""
        if prev_summary:
            user_message += f"""ALREADY GENERATED (for reference):
{prev_summary}

"""

        user_message += self._file_specific_instructions(filepath)

        user_message += """
Output ONLY the raw code. No ```python fences. No explanations.
Use PyTorch. Include type hints and docstrings. Follow PEP 8."""

        prompt = self._build_prompt(system_instruction, user_message)

        raw = await self.call_model(prompt, max_new_tokens=3072, temperature=0.2)

        # Strip any accidental markdown fences
        code = self._strip_fences(raw)
        return code

    @staticmethod
    def _file_specific_instructions(filepath: str) -> str:
        """Return extra instructions specific to certain file types."""
        fname = filepath.rsplit("/", 1)[-1] if "/" in filepath else filepath

        instructions: Dict[str, str] = {
            "model.py": (
                "SPECIFIC INSTRUCTIONS FOR model.py:\n"
                "- Define the main model class inheriting from nn.Module.\n"
                "- Implement __init__ and forward methods.\n"
                "- Include all sub-modules (attention, embeddings, etc.) as "
                "described in the architecture.\n"
                "- Accept a config dict or dataclass in __init__.\n"
                "- Add a model_summary() helper that prints parameter counts.\n"
            ),
            "trainer.py": (
                "SPECIFIC INSTRUCTIONS FOR trainer.py:\n"
                "- Implement a Trainer class with train(), evaluate(), and "
                "save_checkpoint() methods.\n"
                "- Support mixed-precision training with torch.cuda.amp.\n"
                "- Implement learning-rate scheduling.\n"
                "- Log training metrics per epoch.\n"
                "- Accept model, optimizer, dataloader, and config as init args.\n"
            ),
            "dataset_loader.py": (
                "SPECIFIC INSTRUCTIONS FOR dataset_loader.py:\n"
                "- Implement a PyTorch Dataset subclass.\n"
                "- Include a get_dataloaders() factory function.\n"
                "- Support train/val/test splits.\n"
                "- Include basic data preprocessing/transforms.\n"
            ),
            "config.yaml": (
                "SPECIFIC INSTRUCTIONS FOR config.yaml:\n"
                "- Use YAML format (not Python).\n"
                "- Include sections: model, training, data, logging.\n"
                "- Include all hyperparameters from the research summary.\n"
            ),
            "requirements.txt": (
                "SPECIFIC INSTRUCTIONS FOR requirements.txt:\n"
                "- List all Python dependencies with version pins.\n"
                "- Must include: torch, numpy, pyyaml, tqdm, loguru.\n"
                "- Add any dataset-specific libraries.\n"
            ),
            "README.md": (
                "SPECIFIC INSTRUCTIONS FOR README.md:\n"
                "- Include: project title, description, installation, "
                "usage, training, evaluation, project structure.\n"
                "- Reference the original paper.\n"
            ),
            "Dockerfile": (
                "SPECIFIC INSTRUCTIONS FOR Dockerfile:\n"
                "- Base image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime\n"
                "- Copy requirements and install deps first (layer caching).\n"
                "- Copy source code.\n"
                "- Set ENTRYPOINT to training script.\n"
            ),
        }

        return instructions.get(fname, "")

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences if the model wrapped its output."""
        # Remove leading ```<lang> and trailing ```
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()
