"""ReagentAI Research Agent.

The ResearchAgent is the first agent in the pipeline. It receives the raw
text extracted from a research paper and uses Meta Llama-3 to produce a
structured summary covering:

* Algorithm / method name and high-level description
* Architecture components (layers, modules, connections)
* Dataset(s) used for evaluation
* Key hyperparameters
* Training strategy (optimizer, schedule, augmentation, etc.)
* Important equations / formulas

The output dictionary is consumed by downstream agents (PlanningAgent,
ArchitectureAgent, CodingAgent, etc.).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from backend.agents.base_agent import BaseAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger


class ResearchAgent(BaseAgent):
    """Extract structured information from a research paper.

    Uses the Llama-3 Instruct model to parse free-form academic text and
    return a well-defined JSON dictionary that downstream agents can
    consume deterministically.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ResearchAgent",
            model_id=settings.research_agent_model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse the paper text and return a structured research summary.

        Args:
            context: Must contain:
                - ``paper_text`` (str): The full or chunked paper content.
                Optionally:
                - ``paper_title`` (str): Title of the paper if known.

        Returns:
            A dictionary with the following keys:
                ``algorithm``, ``architecture_components``, ``datasets``,
                ``hyperparameters``, ``training_strategy``, ``equations``,
                ``summary``, ``raw_model_output``.
        """
        paper_text: str = context.get("paper_text", "")
        paper_title: str = context.get("paper_title", "Untitled Paper")

        if not paper_text.strip():
            self.logger.error("No paper_text provided in context")
            return self._empty_result("No paper text supplied.")

        self.log_reasoning("start", f"Analysing paper: {paper_title}")

        # Truncate very long papers to fit within model context window
        max_chars = 12_000
        truncated_text = paper_text[:max_chars]
        if len(paper_text) > max_chars:
            self.log_reasoning(
                "truncation",
                f"Paper truncated from {len(paper_text)} to {max_chars} characters",
            )

        # ----- Build the prompt -----
        system_instruction = (
            "You are an expert machine-learning research analyst. "
            "Your task is to read the provided research paper excerpt and "
            "extract structured information. Respond ONLY with a valid JSON "
            "object -- no markdown fences, no commentary."
        )

        user_message = f"""Paper title: {paper_title}

--- BEGIN PAPER TEXT ---
{truncated_text}
--- END PAPER TEXT ---

Extract the following information and return a JSON object with exactly these keys:

{{
  "algorithm": "<Name and one-paragraph description of the core algorithm or method>",
  "architecture_components": [
    "<component 1: e.g. 'Multi-Head Self-Attention layer'>",
    "<component 2: e.g. 'Feed-Forward Network with GELU activation'>",
    ...
  ],
  "datasets": [
    {{
      "name": "<dataset name>",
      "description": "<brief description, size, modality>"
    }}
  ],
  "hyperparameters": {{
    "learning_rate": "<value or range>",
    "batch_size": "<value>",
    "epochs": "<value>",
    "optimizer": "<name>",
    "other": {{}}
  }},
  "training_strategy": "<Paragraph describing training procedure: optimizer, LR schedule, augmentation, regularisation, etc.>",
  "equations": [
    {{
      "name": "<equation label>",
      "latex": "<LaTeX representation>",
      "description": "<plain-English explanation>"
    }}
  ],
  "summary": "<A concise 3-5 sentence summary of the paper's contribution>"
}}

Important:
- If a field is not mentioned in the paper, use "Not specified" or an empty list.
- Return ONLY the JSON object, nothing else."""

        prompt = self._build_prompt(system_instruction, user_message)

        # ----- Call the model -----
        self.log_reasoning("model_call", "Sending extraction prompt to LLM")
        raw_output = await self.call_model(prompt, max_new_tokens=3072, temperature=0.2)
        self.log_reasoning("model_response", f"Received {len(raw_output)} chars")

        # ----- Parse the JSON response -----
        result = self._parse_model_output(raw_output)
        result["raw_model_output"] = raw_output

        self.log_reasoning(
            "complete",
            f"Extracted algorithm: {result.get('algorithm', 'N/A')[:80]}",
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_model_output(self, raw: str) -> Dict[str, Any]:
        """Attempt to parse the model's raw text as JSON.

        If direct parsing fails the method tries to locate a JSON object
        within the text using a simple brace-matching heuristic.

        Args:
            raw: Raw text returned by the language model.

        Returns:
            Parsed dictionary, or a fallback structure on failure.
        """
        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON block from surrounding text
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        self.logger.warning("Failed to parse model output as JSON; returning raw text")
        return self._empty_result(raw)

    @staticmethod
    def _empty_result(reason: str = "") -> Dict[str, Any]:
        """Return a skeleton result dict when extraction fails."""
        return {
            "algorithm": reason or "Extraction failed",
            "architecture_components": [],
            "datasets": [],
            "hyperparameters": {
                "learning_rate": "Not specified",
                "batch_size": "Not specified",
                "epochs": "Not specified",
                "optimizer": "Not specified",
                "other": {},
            },
            "training_strategy": "Not specified",
            "equations": [],
            "summary": reason or "Could not extract summary.",
            "raw_model_output": "",
        }
