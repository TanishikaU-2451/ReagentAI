"""ReagentAI Diagram Agent.

The DiagramAgent generates Mermaid.js diagram code for visualising the
ML model architecture, training pipeline, and data flow. The diagrams
are returned as strings that the frontend can render using the Mermaid
library.

Diagram types produced:

* **model_architecture** -- A flowchart / block diagram showing the
  layers, modules, and data dimensions of the neural network.
* **training_pipeline** -- A sequence/flowchart showing the end-to-end
  training loop (data load -> forward -> loss -> backward -> optimise).
* **data_flow** -- A diagram depicting how data moves from raw input
  through preprocessing, batching, and into the model.

Output format::

    {
        "diagrams": {
            "model_architecture": {
                "type": "flowchart",
                "title": "Model Architecture",
                "mermaid": "flowchart TD\\n  ..."
            },
            "training_pipeline": { ... },
            "data_flow": { ... }
        }
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from backend.agents.base_agent import BaseAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger


class DiagramAgent(BaseAgent):
    """Generate Mermaid.js diagrams for the ML project.

    The agent reads the research summary and generated source files to
    produce accurate, well-labelled Mermaid diagrams that describe the
    system visually.
    """

    def __init__(self) -> None:
        super().__init__(
            name="DiagramAgent",
            model_id=settings.diagram_agent_model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Mermaid diagrams for the project.

        Args:
            context: Must contain:
                - ``research_summary`` (dict): Output of ResearchAgent.
                Optionally:
                - ``files`` (dict): Generated source files (for accuracy).
                - ``architecture`` (dict): Output of ArchitectureAgent.

        Returns:
            Dictionary with ``diagrams``, where each diagram has
            ``type``, ``title``, and ``mermaid`` keys.
        """
        research: Dict[str, Any] = context.get("research_summary", {})
        source_files: Dict[str, str] = context.get("files", {})
        architecture: Dict[str, Any] = context.get("architecture", {})

        if not research:
            self.logger.error("No research_summary in context")
            return {"diagrams": {}}

        self.log_reasoning("start", "Generating Mermaid diagrams")

        diagrams: Dict[str, Dict[str, str]] = {}

        # --- 1. Model Architecture Diagram ---
        self.log_reasoning("generate_diagram", "model_architecture")
        arch_diagram = await self._generate_model_architecture_diagram(
            research, source_files
        )
        diagrams["model_architecture"] = arch_diagram

        # --- 2. Training Pipeline Diagram ---
        self.log_reasoning("generate_diagram", "training_pipeline")
        pipeline_diagram = await self._generate_training_pipeline_diagram(
            research, source_files
        )
        diagrams["training_pipeline"] = pipeline_diagram

        # --- 3. Data Flow Diagram ---
        self.log_reasoning("generate_diagram", "data_flow")
        data_diagram = await self._generate_data_flow_diagram(
            research, source_files
        )
        diagrams["data_flow"] = data_diagram

        self.log_reasoning("complete", f"Generated {len(diagrams)} diagrams")

        return {"diagrams": diagrams}

    # ------------------------------------------------------------------
    # Diagram generators
    # ------------------------------------------------------------------

    async def _generate_model_architecture_diagram(
        self,
        research: Dict[str, Any],
        source_files: Dict[str, str],
    ) -> Dict[str, str]:
        """Generate a Mermaid flowchart for the model architecture."""
        algorithm = research.get("algorithm", "Not specified")
        components = research.get("architecture_components", [])
        equations = research.get("equations", [])

        # Extract model code snippet if available
        model_snippet = ""
        for fp, code in source_files.items():
            if fp.endswith("model.py"):
                model_snippet = code[:4000]
                break

        components_text = "\n".join(f"  - {c}" for c in components) or "  (none)"
        equations_text = "\n".join(
            f"  - {eq.get('name', 'eq')}: {eq.get('description', '')}"
            if isinstance(eq, dict) else f"  - {eq}"
            for eq in equations
        ) or "  (none)"

        system_instruction = (
            "You are an expert at creating clear, accurate Mermaid.js "
            "diagrams for ML model architectures. Generate a Mermaid "
            "flowchart that shows the neural network layers and data "
            "flow through the model. Output ONLY the raw Mermaid code, "
            "starting with 'flowchart TD' or 'graph TD'. No markdown fences."
        )

        user_message = f"""Algorithm: {algorithm}

Architecture Components:
{components_text}

Key Equations:
{equations_text}

"""
        if model_snippet:
            user_message += f"""Model code (for reference):
{model_snippet}

"""
        user_message += """Generate a Mermaid flowchart diagram for this model architecture.

Requirements:
- Use flowchart TD (top-down) direction.
- Show the input at the top and output at the bottom.
- Label each node with the component name and key parameters (e.g. dimensions).
- Use appropriate Mermaid shapes: [rectangles] for layers, ((circles)) for operations, {{rhombus}} for decisions.
- Connect nodes with labeled arrows showing tensor shapes where useful.
- Keep it readable -- max 20 nodes.
- Output ONLY the Mermaid code, nothing else."""

        prompt = self._build_prompt(system_instruction, user_message)
        raw = await self.call_model(prompt, max_new_tokens=1536, temperature=0.3)
        mermaid_code = self._clean_mermaid(raw)

        return {
            "type": "flowchart",
            "title": "Model Architecture",
            "mermaid": mermaid_code,
        }

    async def _generate_training_pipeline_diagram(
        self,
        research: Dict[str, Any],
        source_files: Dict[str, str],
    ) -> Dict[str, str]:
        """Generate a Mermaid diagram for the training pipeline."""
        training = research.get("training_strategy", "Not specified")
        hyperparams = research.get("hyperparameters", {})

        trainer_snippet = ""
        for fp, code in source_files.items():
            if fp.endswith("trainer.py"):
                trainer_snippet = code[:3000]
                break

        system_instruction = (
            "You are an expert at creating Mermaid.js diagrams for ML "
            "training pipelines. Generate a flowchart showing the complete "
            "training loop. Output ONLY raw Mermaid code. No markdown fences."
        )

        user_message = f"""Training Strategy: {training}

Hyperparameters: {json.dumps(hyperparams, indent=2, default=str)}

"""
        if trainer_snippet:
            user_message += f"""Trainer code (excerpt):
{trainer_snippet}

"""
        user_message += """Generate a Mermaid flowchart for the training pipeline.

Requirements:
- Use flowchart TD direction.
- Include: data loading, forward pass, loss computation, backward pass,
  optimizer step, LR scheduling, checkpointing, logging.
- Show the epoch/batch loop structure.
- Include validation step if applicable.
- Label nodes clearly.
- Max 15 nodes.
- Output ONLY the Mermaid code."""

        prompt = self._build_prompt(system_instruction, user_message)
        raw = await self.call_model(prompt, max_new_tokens=1536, temperature=0.3)
        mermaid_code = self._clean_mermaid(raw)

        return {
            "type": "flowchart",
            "title": "Training Pipeline",
            "mermaid": mermaid_code,
        }

    async def _generate_data_flow_diagram(
        self,
        research: Dict[str, Any],
        source_files: Dict[str, str],
    ) -> Dict[str, str]:
        """Generate a Mermaid diagram for the data flow."""
        datasets = research.get("datasets", [])
        algorithm = research.get("algorithm", "Not specified")

        dataset_snippet = ""
        for fp, code in source_files.items():
            if fp.endswith("dataset_loader.py"):
                dataset_snippet = code[:3000]
                break

        datasets_text = "\n".join(
            f"  - {d['name']}: {d.get('description', '')}"
            if isinstance(d, dict) else f"  - {d}"
            for d in datasets
        ) or "  Not specified"

        system_instruction = (
            "You are an expert at creating Mermaid.js diagrams for data "
            "processing pipelines. Generate a flowchart showing data flow "
            "from raw input to model consumption. Output ONLY raw Mermaid "
            "code. No markdown fences."
        )

        user_message = f"""Algorithm: {algorithm}

Datasets:
{datasets_text}

"""
        if dataset_snippet:
            user_message += f"""Dataset loader code (excerpt):
{dataset_snippet}

"""
        user_message += """Generate a Mermaid flowchart for the data processing pipeline.

Requirements:
- Use flowchart LR (left-right) or TD direction.
- Show: raw data source, preprocessing steps, tokenization/transforms,
  batching via DataLoader, and feeding into the model.
- Include train/val/test split if applicable.
- Label each step with key parameters.
- Max 12 nodes.
- Output ONLY the Mermaid code."""

        prompt = self._build_prompt(system_instruction, user_message)
        raw = await self.call_model(prompt, max_new_tokens=1536, temperature=0.3)
        mermaid_code = self._clean_mermaid(raw)

        return {
            "type": "flowchart",
            "title": "Data Flow",
            "mermaid": mermaid_code,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_mermaid(raw: str) -> str:
        """Clean LLM output to extract valid Mermaid code.

        Strips markdown fences and any explanatory text that may surround
        the diagram code.
        """
        text = raw.strip()

        # Remove markdown mermaid fences
        text = re.sub(r"^```mermaid\s*\n?", "", text)
        text = re.sub(r"^```\w*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

        # If there is explanatory text before the diagram keyword, strip it
        for keyword in ("flowchart", "graph", "sequenceDiagram", "classDiagram"):
            idx = text.find(keyword)
            if idx > 0:
                text = text[idx:]
                break

        return text.strip()
