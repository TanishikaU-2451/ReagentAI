"""ReagentAI Chat Agent.

The ChatAgent handles interactive user questions about the research paper,
the generated implementation, or machine-learning concepts in general. It
uses Retrieval-Augmented Generation (RAG) by incorporating relevant chunks
from the paper text and generated source files into its prompt.

The agent is designed to be invoked repeatedly -- once per user message --
and it maintains conversational context through the ``history`` list
passed in via the context dictionary.

Output format::

    {
        "answer": "The attention mechanism in this paper works by ...",
        "sources": ["paper chunk 3", "src/model.py lines 42-60"],
        "follow_up_suggestions": [
            "How does the positional encoding work?",
            "What loss function is used?"
        ]
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from backend.agents.base_agent import BaseAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger


class ChatAgent(BaseAgent):
    """Answer user questions about the paper and implementation.

    The agent combines RAG context (paper chunks, generated code) with
    the user's question and conversation history to produce a grounded,
    accurate answer.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ChatAgent",
            model_id=settings.chat_agent_model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Answer a user question using RAG context.

        Args:
            context: Must contain:
                - ``question`` (str): The user's question.
                Optionally:
                - ``rag_chunks`` (list[str]): Retrieved paper chunks.
                - ``files`` (dict): Generated source files for code context.
                - ``research_summary`` (dict): Structured paper summary.
                - ``history`` (list[dict]): Conversation history, each dict
                  having ``role`` (user/assistant) and ``content`` keys.

        Returns:
            Dictionary with ``answer``, ``sources``, and
            ``follow_up_suggestions``.
        """
        question: str = context.get("question", "")
        rag_chunks: List[str] = context.get("rag_chunks", [])
        source_files: Dict[str, str] = context.get("files", {})
        research: Dict[str, Any] = context.get("research_summary", {})
        history: List[Dict[str, str]] = context.get("history", [])

        if not question.strip():
            self.logger.error("Empty question provided")
            return {
                "answer": "Please provide a question.",
                "sources": [],
                "follow_up_suggestions": [],
            }

        self.log_reasoning("start", f"Answering: {question[:100]}")

        # ----- Build RAG context block -----
        rag_context = self._build_rag_context(
            rag_chunks=rag_chunks,
            source_files=source_files,
            research=research,
            question=question,
        )

        # ----- Build conversation history block -----
        history_text = self._format_history(history)

        # ----- Build prompt -----
        system_instruction = (
            "You are ReagentAI, a helpful assistant that answers questions "
            "about machine-learning research papers and their implementations. "
            "You have access to excerpts from the paper and generated source "
            "code. Base your answers on the provided context. If the context "
            "does not contain enough information, say so honestly. "
            "Be concise, accurate, and cite your sources when possible.\n\n"
            "After your answer, suggest 2-3 follow-up questions the user "
            "might find useful.\n\n"
            "Respond ONLY with a JSON object with keys: "
            '"answer", "sources", "follow_up_suggestions".'
        )

        user_message = f"""CONTEXT:
{rag_context}

{history_text}

USER QUESTION: {question}

Respond with a JSON object:
{{
  "answer": "<Your detailed answer, using information from the context. Use markdown formatting for clarity.>",
  "sources": ["<source 1: e.g. 'Paper section 3.2'>", "<source 2: e.g. 'src/model.py'>"],
  "follow_up_suggestions": [
    "<suggested question 1>",
    "<suggested question 2>",
    "<suggested question 3>"
  ]
}}

Return ONLY the JSON object."""

        prompt = self._build_prompt(system_instruction, user_message)

        # ----- Call model -----
        self.log_reasoning("model_call", "Sending question to ChatAgent LLM")
        raw_output = await self.call_model(prompt, max_new_tokens=2048, temperature=0.4)
        self.log_reasoning("model_response", f"Received {len(raw_output)} chars")

        # ----- Parse response -----
        result = self._parse_response(raw_output, question)

        self.log_reasoning(
            "complete",
            f"Answer length: {len(result.get('answer', ''))} chars, "
            f"sources: {len(result.get('sources', []))}",
        )
        return result

    # ------------------------------------------------------------------
    # Context building helpers
    # ------------------------------------------------------------------

    def _build_rag_context(
        self,
        rag_chunks: List[str],
        source_files: Dict[str, str],
        research: Dict[str, Any],
        question: str,
    ) -> str:
        """Assemble the RAG context block for the prompt.

        Combines paper chunks, the research summary, and relevant source
        file snippets into a single context string.

        Args:
            rag_chunks:   Retrieved paper text chunks.
            source_files: Generated project files.
            research:     Structured research summary.
            question:     User question (used for relevance heuristics).

        Returns:
            Formatted context string.
        """
        parts: List[str] = []

        # Research summary
        if research:
            parts.append("=== PAPER SUMMARY ===")
            parts.append(f"Algorithm: {research.get('algorithm', 'N/A')}")
            parts.append(f"Summary: {research.get('summary', 'N/A')}")
            components = research.get("architecture_components", [])
            if components:
                parts.append("Components: " + ", ".join(components[:10]))
            parts.append(f"Training: {research.get('training_strategy', 'N/A')}")
            parts.append("")

        # RAG chunks from the paper
        if rag_chunks:
            parts.append("=== PAPER EXCERPTS ===")
            for i, chunk in enumerate(rag_chunks[:5], 1):
                trimmed = chunk[:1500]
                parts.append(f"[Chunk {i}]\n{trimmed}\n")
            parts.append("")

        # Relevant source file snippets
        if source_files:
            relevant_files = self._select_relevant_files(source_files, question)
            if relevant_files:
                parts.append("=== SOURCE CODE EXCERPTS ===")
                for fp, snippet in relevant_files.items():
                    parts.append(f"--- {fp} ---\n{snippet}\n")

        context = "\n".join(parts)
        # Cap total context length
        return context[:8000]

    def _select_relevant_files(
        self,
        source_files: Dict[str, str],
        question: str,
    ) -> Dict[str, str]:
        """Heuristically select source files most relevant to the question.

        Uses keyword matching to rank files by relevance.

        Args:
            source_files: All generated source files.
            question:     The user's question.

        Returns:
            Dict of filepath -> code snippet for the top relevant files.
        """
        question_lower = question.lower()

        # Keyword-to-file relevance map
        relevance_keywords: Dict[str, List[str]] = {
            "model": ["model", "architecture", "layer", "attention", "network", "forward"],
            "trainer": ["train", "loss", "optimizer", "epoch", "learning rate", "backward"],
            "dataset_loader": ["data", "dataset", "loader", "batch", "preprocess", "transform"],
            "config": ["config", "hyperparameter", "parameter", "setting"],
        }

        scored: List[tuple[str, str, int]] = []
        for fp, code in source_files.items():
            if not fp.endswith(".py"):
                continue
            fname = fp.rsplit("/", 1)[-1].replace(".py", "")
            score = 0
            keywords = relevance_keywords.get(fname, [])
            for kw in keywords:
                if kw in question_lower:
                    score += 2
            # Also check if the filename appears in the question
            if fname in question_lower:
                score += 3

            scored.append((fp, code, score))

        # Sort by score descending, take top 3
        scored.sort(key=lambda x: x[2], reverse=True)
        result: Dict[str, str] = {}
        for fp, code, score in scored[:3]:
            if score > 0:
                result[fp] = code[:2000]
            elif not result:
                # Always include at least model.py
                result[fp] = code[:2000]

        return result

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        """Format conversation history for the prompt.

        Args:
            history: List of ``{"role": "user"|"assistant", "content": "..."}``

        Returns:
            Formatted history string, or empty string if no history.
        """
        if not history:
            return ""

        lines = ["=== CONVERSATION HISTORY ==="]
        # Include last 6 turns to stay within context limits
        for turn in history[-6:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")[:500]
            lines.append(f"{role}: {content}")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, question: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured answer dict."""
        # Try direct JSON
        try:
            parsed = json.loads(raw)
            if "answer" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # Try brace extraction
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                parsed = json.loads(match.group())
                if "answer" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: treat the whole output as the answer
        self.logger.warning("Could not parse chat response as JSON; using raw text")
        return {
            "answer": raw.strip(),
            "sources": [],
            "follow_up_suggestions": [],
        }
