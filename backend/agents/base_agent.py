
"""ReagentAI Base Agent.



Provides the abstract base class for all agents in the multi-agent system.
Every agent inherits from BaseAgent and implements the ``run`` method, which
receives an arbitrary context dictionary and returns a result dictionary.

The shared ``call_model`` helper handles communication with the HuggingFace
Inference API, including retries, rate-limit back-off, and error handling.
"""

from __future__ import annotations

import abc
import asyncio
import time
from typing import Any, Dict, Optional

import httpx

from backend.config.settings import settings
from backend.utils.logging import get_logger

# HuggingFace Inference API base URL (New OpenAI-compatible endpoint)
HF_INFERENCE_URL = "https://router.huggingface.co/v1/chat/completions"

# Retry / rate-limit defaults
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 2.0
_REQUEST_TIMEOUT_SECONDS = 120.0


class BaseAgent(abc.ABC):
    """Abstract base class for all ReagentAI agents.

    Attributes:
        name:     Human-readable agent name (e.g. ``"ResearchAgent"``).
        model_id: HuggingFace model identifier used by this agent.
        logger:   Loguru logger bound to the agent name.
    """

    def __init__(self, name: str, model_id: str) -> None:
        """Initialise the agent.

        Args:
            name:     A descriptive name for the agent instance.
            model_id: The HuggingFace Hub model identifier
                      (e.g. ``"meta-llama/Meta-Llama-3-8B-Instruct"``).
        """
        self.name: str = name
        self.model_id: str = model_id
        self.logger = get_logger(f"agent.{name}")
        self.logger.info(
            "Initialised {name} with model {model}", name=name, model=model_id
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main task.

        Each concrete agent must override this method with its own logic.
        The *context* dictionary carries whatever upstream data the agent
        needs (e.g. extracted text, previous agent outputs).

        Args:
            context: Arbitrary data consumed by the agent.

        Returns:
            A dictionary containing the agent's output.
        """
        ...

    # ------------------------------------------------------------------
    # Model invocation
    # ------------------------------------------------------------------

    async def call_model(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.9,
        stop_sequences: Optional[list[str]] = None,
    ) -> str:
        """Call the HuggingFace Inference API and return generated text.

        Uses the new OpenAI-compatible API endpoint with automatic provider selection.

        Args:
            prompt:          The full prompt string to send.
            max_new_tokens:  Maximum tokens to generate.
            temperature:     Sampling temperature.
            top_p:           Nucleus sampling probability.
            stop_sequences:  Optional list of stop strings.

        Returns:
            The generated text as a string.

        Raises:
            httpx.HTTPStatusError: If the request fails after all retries.
        """
        headers = {
            "Authorization": f"Bearer {settings.huggingface_api_token}",
            "Content-Type": "application/json",
        }

        # Use OpenAI-compatible format with automatic provider selection
        payload = {
            "model": f"{self.model_id}:fastest",  # Use :fastest for best performance
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        self.log_reasoning("call_model", f"Sending request to {self.model_id}")

        backoff = _INITIAL_BACKOFF_SECONDS

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    response = await client.post(HF_INFERENCE_URL, headers=headers, json=payload)

                    # Model still loading -- wait and retry
                    if response.status_code == 503:
                        self.logger.warning(
                            "Model loading (503). Retry {attempt}/{max} in {wait}s",
                            attempt=attempt,
                            max=_MAX_RETRIES,
                            wait=round(backoff, 1),
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    # Rate-limited
                    if response.status_code == 429:
                        self.logger.warning(
                            "Rate-limited (429). Retry {attempt}/{max} in {wait}s",
                            attempt=attempt,
                            max=_MAX_RETRIES,
                            wait=round(backoff, 1),
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    response.raise_for_status()

                    data = response.json()

                    # Parse OpenAI-compatible response format
                    generated = ""
                    if isinstance(data, dict) and "choices" in data:
                        if len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                generated = choice["message"]["content"]
                            elif "text" in choice:  # Fallback for some providers
                                generated = choice["text"]

                    # Fallback parsing for different response formats
                    if not generated and isinstance(data, dict):
                        generated = data.get("generated_text", data.get("content", ""))

                    if not generated:
                        generated = str(data)

                    # Validate response is not empty
                    if not generated.strip():
                        self.logger.warning(
                            "Model returned empty response on attempt {}/{}",
                            attempt,
                            _MAX_RETRIES
                        )
                        if attempt < _MAX_RETRIES:
                            await asyncio.sleep(backoff)
                            backoff *= 2
                            continue
                        else:
                            return "Error: Model returned empty response after all retries."

                    self.log_reasoning(
                        "call_model",
                        f"Received {len(generated)} characters from {self.model_id}",
                    )
                    return generated.strip()

                except httpx.ReadTimeout:
                    self.logger.warning(
                        "Timeout on attempt {attempt}/{max}",
                        attempt=attempt,
                        max=_MAX_RETRIES,
                    )
                    if attempt == _MAX_RETRIES:
                        raise
                    await asyncio.sleep(backoff)
                    backoff *= 2

                except httpx.HTTPStatusError as exc:
                    self.logger.error(
                        "HTTP {status} from {model}: {detail}",
                        status=exc.response.status_code,
                        model=self.model_id,
                        detail=exc.response.text[:500],
                    )
                    raise

        # Should not reach here, but just in case
        raise RuntimeError(
            f"{self.name}: all {_MAX_RETRIES} retries exhausted for {self.model_id}"
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_reasoning(self, step: str, detail: str) -> None:
        """Log an agent reasoning step.

        Produces a structured log entry so that agent decision-making can
        be traced in the ``agents.log`` file.

        Args:
            step:   Short label for the reasoning phase (e.g. ``"parse"``).
            detail: Free-form description of what the agent decided / observed.
        """
        self.logger.info(
            "[{agent}] {step} -- {detail}",
            agent=self.name,
            step=step,
            detail=detail,
        )

    def _build_prompt(self, system: str, user: str) -> str:
        """Build a simple instruction-style prompt.

        This helper concatenates a system instruction and a user message
        into a single prompt string compatible with most instruction-tuned
        models on the HuggingFace Hub.

        Uses a model-agnostic format that works with Flan-T5, DialoGPT, and similar models.

        Args:
            system: System-level instruction.
            user:   User-level request / data.

        Returns:
            A formatted prompt string.
        """
        # Simple format compatible with Flan-T5 and DialoGPT
        return f"{system}\n\n{user}"
