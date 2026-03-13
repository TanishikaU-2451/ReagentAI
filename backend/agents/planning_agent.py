"""ReagentAI Planning Agent.

The PlanningAgent converts the structured research summary produced by the
ResearchAgent into an ordered implementation plan. It uses Mixtral to reason
about dependencies between components and to produce a step-by-step checklist
that the CodingAgent can follow sequentially.

Output format::

    {
        "steps": [
            "Step 1: Implement the data loading pipeline ...",
            "Step 2: Implement the embedding layer ...",
            ...
        ],
        "dependencies": { "step_2": ["step_1"], ... },
        "estimated_files": ["model.py", "trainer.py", ...],
        "notes": "Any additional planning notes."
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from backend.agents.base_agent import BaseAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger


class PlanningAgent(BaseAgent):
    """Convert a research summary into a concrete implementation plan.

    The agent analyses the algorithm description, architecture components,
    training strategy, and hyperparameters to determine the most logical
    implementation order and to anticipate which source files will be
    needed.
    """

    def __init__(self) -> None:
        super().__init__(
            name="PlanningAgent",
            model_id=settings.planning_agent_model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an implementation plan from a research summary.

        Args:
            context: Must contain:
                - ``research_summary`` (dict): Output of ResearchAgent.

        Returns:
            A dictionary with keys ``steps``, ``dependencies``,
            ``estimated_files``, and ``notes``.
        """
        research: Dict[str, Any] = context.get("research_summary", {})
        if not research:
            self.logger.error("No research_summary provided in context")
            return self._empty_plan("No research summary available.")

        self.log_reasoning("start", "Building implementation plan from research summary")

        # ----- Serialise the research summary for the prompt -----
        algorithm = research.get("algorithm", "Not specified")
        components = research.get("architecture_components", [])
        datasets = research.get("datasets", [])
        hyperparams = research.get("hyperparameters", {})
        training = research.get("training_strategy", "Not specified")
        equations = research.get("equations", [])
        summary = research.get("summary", "")

        components_text = "\n".join(f"  - {c}" for c in components) or "  (none listed)"
        datasets_text = "\n".join(
            f"  - {d['name']}: {d.get('description', '')}"
            if isinstance(d, dict) else f"  - {d}"
            for d in datasets
        ) or "  (none listed)"
        equations_text = "\n".join(
            f"  - {eq.get('name', 'eq')}: {eq.get('description', '')}"
            if isinstance(eq, dict) else f"  - {eq}"
            for eq in equations
        ) or "  (none listed)"

        hyperparams_text = json.dumps(hyperparams, indent=2, default=str)

        # ----- Build prompt -----
        system_instruction = (
            "You are a senior ML engineer who specialises in converting "
            "research papers into clean, modular PyTorch implementations. "
            "Given a structured research summary, produce a detailed "
            "step-by-step implementation plan. Respond ONLY with valid JSON."
        )

        user_message = f"""Research Summary
================
Algorithm: {algorithm}

Architecture Components:
{components_text}

Datasets:
{datasets_text}

Hyperparameters:
{hyperparams_text}

Training Strategy: {training}

Key Equations:
{equations_text}

Paper Summary: {summary}

---

Create a detailed implementation plan as a JSON object with these keys:

{{
  "steps": [
    "Step 1: <action> -- <rationale>",
    "Step 2: ...",
    ...
  ],
  "dependencies": {{
    "step_2": ["step_1"],
    ...
  }},
  "estimated_files": [
    "model.py",
    "trainer.py",
    ...
  ],
  "notes": "<Any additional notes about implementation order or gotchas>"
}}

Guidelines for the plan:
1. Start with configuration and data loading.
2. Implement model components bottom-up (lowest-level modules first).
3. Implement the full model by composing sub-modules.
4. Implement the training loop, loss functions, and metrics.
5. Add evaluation / inference scripts.
6. Add Dockerfile, requirements.txt, README, and config.yaml.
7. Each step should be specific enough to map to a single function or class.
8. Include at least 8 steps and at most 20 steps.

Return ONLY the JSON object."""

        prompt = self._build_prompt(system_instruction, user_message)

        # ----- Call model -----
        self.log_reasoning("model_call", "Requesting implementation plan from Mixtral")
        raw_output = await self.call_model(prompt, max_new_tokens=2048, temperature=0.3)
        self.log_reasoning("model_response", f"Received {len(raw_output)} chars")

        # ----- Parse response -----
        result = self._parse_plan(raw_output)
        result["raw_model_output"] = raw_output

        self.log_reasoning(
            "complete",
            f"Plan contains {len(result.get('steps', []))} steps",
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_plan(self, raw: str) -> Dict[str, Any]:
        """Parse the model output into a plan dictionary."""
        # Direct JSON parse
        try:
            parsed = json.loads(raw)
            if "steps" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # Attempt brace extraction
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                parsed = json.loads(match.group())
                if "steps" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: try to find numbered steps line by line
        self.logger.warning("JSON parse failed; falling back to line extraction")
        steps: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if re.match(r"^(Step\s+)?\d+[\.\):]", line):
                steps.append(line)

        if steps:
            return {
                "steps": steps,
                "dependencies": {},
                "estimated_files": [],
                "notes": "Plan extracted via fallback line parser.",
            }

        return self._empty_plan("Could not parse planning output.")

    @staticmethod
    def _empty_plan(reason: str = "") -> Dict[str, Any]:
        """Return a skeleton plan when generation fails."""
        return {
            "steps": [reason or "Planning failed"],
            "dependencies": {},
            "estimated_files": [],
            "notes": reason,
            "raw_model_output": "",
        }
