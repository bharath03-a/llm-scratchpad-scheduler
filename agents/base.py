"""Base async Gemini agent with retry logic and robust JSON extraction."""
from __future__ import annotations

import json
import re
import sys
from typing import Any

import google.genai as genai


class BaseAgent:
    """
    Shared foundation for all scheduling agents.
    - Async Gemini calls via client.aio
    - Robust JSON extraction (handles markdown fences, extra prose)
    - Configurable retry logic
    """

    DEFAULT_MAX_RETRIES = 3

    def __init__(self, client: genai.Client, model_name: str = "gemini-2.5-flash") -> None:
        self.client = client
        self.model_name = model_name

    # ------------------------------------------------------------------
    # JSON extraction  (BUG-4 fix: handles markdown-wrapped responses)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_json(text: str) -> dict[str, Any] | None:
        """
        Robustly extract a JSON object from an LLM response.
        Handles:
          - Raw JSON
          - ```json ... ``` fences
          - ``` ... ``` fences
          - Prose before/after the JSON object
        """
        if not text:
            return None

        # 1. Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
        stripped = re.sub(r"```(?:json)?\s*", "", text)
        stripped = re.sub(r"```\s*", "", stripped).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # 3. Find the first {...} block (handles prose wrapping)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    # Async LLM call helpers
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_output: bool = True,
        label: str = "",
        thinking_budget: int = 0,
    ) -> str:
        """Make an async LLM call with retry logic.

        thinking_budget: Token budget for model thinking (0 = disabled).
        Gemini 2.5 Flash counts thinking tokens against max_output_tokens, so
        disabling thinking frees the full budget for actual JSON/text output.
        """
        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "thinking_config": {"thinking_budget": thinking_budget},
        }
        if json_output:
            config["response_mime_type"] = "application/json"

        for attempt in range(self.DEFAULT_MAX_RETRIES):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                text = response.text or ""
                if not text:
                    candidate = response.candidates[0] if response.candidates else None
                    finish = getattr(candidate, "finish_reason", "?")
                    tag = f"[{label}] " if label else ""
                    print(f"{tag}Attempt {attempt + 1}: empty response — finish_reason={finish}", file=sys.stderr)
                    if attempt < self.DEFAULT_MAX_RETRIES - 1:
                        continue
                return text
            except Exception as exc:
                tag = f"[{label}] " if label else ""
                print(f"{tag}Attempt {attempt + 1}/{self.DEFAULT_MAX_RETRIES} failed: {exc}", file=sys.stderr)
                if attempt == self.DEFAULT_MAX_RETRIES - 1:
                    raise
        return ""

    async def _call_llm_text(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        label: str = "",
        thinking_budget: int = 4096,
    ) -> str:
        """Make an async LLM call expecting free-text (not JSON).

        Uses a moderate thinking budget by default — useful for analysis tasks
        that benefit from reasoning before producing text.
        """
        return await self._call_llm(
            prompt, temperature, max_tokens,
            json_output=False, label=label, thinking_budget=thinking_budget,
        )
