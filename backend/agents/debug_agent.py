"""ReagentAI Debug Agent.

The DebugAgent analyses error output from failed tests or execution runs,
cross-references the error with the relevant source code, and uses
CodeLlama to diagnose the problem and propose a corrected version of the
offending file(s).

It is designed to be called inside a retry loop controlled by the
orchestrator. Each call receives the latest error output and the current
source files, and returns corrected files plus an explanation.

Output format::

    {
        "diagnosis": "Explanation of the root cause...",
        "corrected_files": {
            "src/model.py": "<fixed code>",
            ...
        },
        "changes_summary": [
            "Fixed shape mismatch in Attention.forward() ...",
            ...
        ],
        "confidence": 0.85,
        "attempt": 1
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from backend.agents.base_agent import BaseAgent
from backend.config.settings import settings
from backend.utils.logging import get_logger


class DebugAgent(BaseAgent):
    """Analyse errors and produce corrected source code.

    The agent performs a three-phase process:

    1. **Diagnose** -- parse the error traceback to identify the failing
       file, line, and error type.
    2. **Reason** -- use the LLM to understand the root cause given the
       source code context.
    3. **Fix** -- generate corrected file content that resolves the issue.
    """

    def __init__(self) -> None:
        super().__init__(
            name="DebugAgent",
            model_id=settings.debug_agent_model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse an error and return corrected code.

        Args:
            context: Must contain:
                - ``error_output`` (str): Full stderr / traceback.
                - ``files`` (dict): Current source files (path -> code).
                Optionally:
                - ``attempt`` (int): Current debug attempt number.
                - ``previous_fixes`` (list[str]): Summaries of prior fix
                  attempts that failed.
                - ``research_summary`` (dict): For additional context.

        Returns:
            Dictionary with ``diagnosis``, ``corrected_files``,
            ``changes_summary``, ``confidence``, and ``attempt``.
        """
        error_output: str = context.get("error_output", "")
        source_files: Dict[str, str] = context.get("files", {})
        attempt: int = context.get("attempt", 1)
        previous_fixes: List[str] = context.get("previous_fixes", [])
        research: Dict[str, Any] = context.get("research_summary", {})

        if not error_output:
            self.logger.error("No error_output provided in context")
            return self._empty_debug_result("No error output supplied.", attempt)

        self.log_reasoning("start", f"Debug attempt {attempt}: analysing error")

        # Phase 1: Extract failing file(s) from the traceback
        suspect_files = self._extract_suspect_files(error_output, source_files)
        self.log_reasoning(
            "traceback_analysis",
            f"Suspect files: {list(suspect_files.keys()) or '(could not determine)'}",
        )

        # Phase 2 + 3: Diagnose and fix via the LLM
        result = await self._diagnose_and_fix(
            error_output=error_output,
            suspect_files=suspect_files,
            all_files=source_files,
            attempt=attempt,
            previous_fixes=previous_fixes,
            research=research,
        )

        result["attempt"] = attempt

        self.log_reasoning(
            "complete",
            f"Diagnosis: {result.get('diagnosis', '')[:120]}",
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_suspect_files(
        self,
        error_output: str,
        source_files: Dict[str, str],
    ) -> Dict[str, str]:
        """Identify which source files are referenced in the traceback.

        Args:
            error_output: Raw traceback / error text.
            source_files: All project source files.

        Returns:
            Subset of source_files that appear in the traceback.
        """
        suspects: Dict[str, str] = {}

        # Look for 'File "..." , line N' patterns
        file_refs = re.findall(r'File\s+"([^"]+)"', error_output)

        for ref in file_refs:
            # Normalise path separators
            ref_norm = ref.replace("\\", "/")
            for fp, code in source_files.items():
                fp_norm = fp.replace("\\", "/")
                if ref_norm.endswith(fp_norm) or fp_norm.endswith(ref_norm.split("/")[-1]):
                    suspects[fp] = code

        # If we couldn't find specific files, include any .py files mentioned
        if not suspects:
            for fp, code in source_files.items():
                fname = fp.rsplit("/", 1)[-1]
                if fname in error_output:
                    suspects[fp] = code

        # Fallback: include model.py and trainer.py as most common culprits
        if not suspects:
            for fp in ["src/model.py", "src/trainer.py"]:
                if fp in source_files:
                    suspects[fp] = source_files[fp]

        return suspects

    async def _diagnose_and_fix(
        self,
        error_output: str,
        suspect_files: Dict[str, str],
        all_files: Dict[str, str],
        attempt: int,
        previous_fixes: List[str],
        research: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use the LLM to diagnose the error and produce fixed files."""
        # Prepare suspect file content (trim if very long)
        suspect_text_parts = []
        for fp, code in suspect_files.items():
            trimmed = code[:5000]
            suspect_text_parts.append(f"--- {fp} ---\n{trimmed}")
        suspect_text = "\n\n".join(suspect_text_parts)

        # Prepare previous-fix context
        prev_fixes_text = ""
        if previous_fixes:
            prev_fixes_text = (
                "\nPREVIOUS FIX ATTEMPTS THAT FAILED:\n"
                + "\n".join(f"  - Attempt: {f}" for f in previous_fixes)
                + "\nDo NOT repeat these fixes. Try a different approach.\n"
            )

        # Algorithm context
        algorithm = research.get("algorithm", "Not specified") if research else "Not specified"

        system_instruction = (
            "You are an expert ML debugging engineer. You are given an error "
            "traceback and the relevant source code. Your job is to:\n"
            "1. Diagnose the ROOT CAUSE of the error.\n"
            "2. Produce the COMPLETE corrected file(s).\n"
            "3. Explain what you changed and why.\n\n"
            "Respond with a JSON object. Do NOT use markdown fences around "
            "the JSON."
        )

        user_message = f"""ERROR OUTPUT:
```
{error_output[:4000]}
```

ALGORITHM CONTEXT: {algorithm}

SUSPECT SOURCE FILES:
{suspect_text}

DEBUG ATTEMPT: {attempt} of {settings.max_debug_attempts}
{prev_fixes_text}

Respond with a JSON object:
{{
  "diagnosis": "<Clear explanation of the root cause>",
  "corrected_files": {{
    "<filepath>": "<COMPLETE corrected file content -- not a diff, the FULL file>"
  }},
  "changes_summary": [
    "<change 1: what was wrong and what you fixed>",
    "<change 2: ...>"
  ],
  "confidence": <float between 0.0 and 1.0>
}}

IMPORTANT:
- corrected_files must contain the COMPLETE file content, not a patch.
- Only include files that actually need changes.
- If the error is in a test file, fix the source file if the logic is wrong,
  or fix the test if the test expectation is wrong.
- Do NOT wrap the JSON in markdown code fences."""

        prompt = self._build_prompt(system_instruction, user_message)

        self.log_reasoning("model_call", f"Sending debug prompt (attempt {attempt})")
        raw_output = await self.call_model(prompt, max_new_tokens=4096, temperature=0.15)
        self.log_reasoning("model_response", f"Received {len(raw_output)} chars")

        return self._parse_debug_output(raw_output, attempt)

    def _parse_debug_output(self, raw: str, attempt: int) -> Dict[str, Any]:
        """Parse the LLM debug response into a structured dict."""
        # Try direct JSON parse
        try:
            parsed = json.loads(raw)
            if "diagnosis" in parsed and "corrected_files" in parsed:
                # Strip any code fences from file contents
                for fp in list(parsed["corrected_files"].keys()):
                    parsed["corrected_files"][fp] = self._strip_fences(
                        parsed["corrected_files"][fp]
                    )
                return parsed
        except json.JSONDecodeError:
            pass

        # Try brace extraction
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                parsed = json.loads(match.group())
                if "diagnosis" in parsed:
                    for fp in list(parsed.get("corrected_files", {}).keys()):
                        parsed["corrected_files"][fp] = self._strip_fences(
                            parsed["corrected_files"][fp]
                        )
                    return parsed
            except json.JSONDecodeError:
                pass

        self.logger.warning("Could not parse debug output as JSON")
        return self._empty_debug_result(
            f"Parse failure. Raw output: {raw[:500]}", attempt
        )

    @staticmethod
    def _empty_debug_result(reason: str, attempt: int) -> Dict[str, Any]:
        """Return a fallback debug result."""
        return {
            "diagnosis": reason,
            "corrected_files": {},
            "changes_summary": [],
            "confidence": 0.0,
            "attempt": attempt,
        }

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences."""
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()
