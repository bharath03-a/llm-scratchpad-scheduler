"""
Solution Refiner Agent.
Given a solution that failed validation, and the specific error list,
makes targeted corrections to produce a valid schedule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from agents.base import BaseAgent

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "system_refinement.txt").read_text()
_MAX_ITERATIONS = 3


class RefinerAgent(BaseAgent):

    async def refine(
        self,
        problem: dict[str, Any],
        solution: dict[str, Any],
        errors: list[str],
    ) -> dict[str, Any] | None:
        solution_str = json.dumps(solution, indent=2)
        errors_str = "\n".join(f"  - {e}" for e in errors)
        problem_summary = _format_problem_summary(problem)

        prompt = f"""{_SYSTEM_PROMPT}

{problem_summary}

## Current (Invalid) Solution:

{solution_str}

## Validation Errors to Fix:

{errors_str}

## Fix Instructions

1. Address EVERY error listed above.
2. Preserve as much of the current solution structure as possible.
3. Do not introduce new errors (re-check all constraints after fixing).
4. Constraints:
   - All {len(problem['op_types'])} operations must appear exactly once.
   - Granularities must be [w,h,k] positive integers.
   - Working set ≤ {problem['fast_memory_capacity']} bytes per step.
   - Dependency order: producer subgraph step ≤ consumer subgraph step.
5. Output ONLY valid JSON, no markdown.
"""
        num_ops = len(problem["op_types"])
        max_tokens = max(8192, min(65536, num_ops * 1000))
        print(
            f"  [Refiner] Fixing {len(errors)} error(s) (max_tokens={max_tokens}, thinking=off)...",
            file=sys.stderr,
        )
        text = await self._call_llm(
            prompt,
            temperature=0.1,
            max_tokens=max_tokens,
            json_output=True,
            label="Refiner",
            thinking_budget=0,
        )
        result = self.extract_json(text)
        if result:
            print("  [Refiner] Done.", file=sys.stderr)
        else:
            print("  [Refiner] Parse failed.", file=sys.stderr)
        return result


def _format_problem_summary(problem: dict[str, Any]) -> str:
    lines = [
        "## Problem Summary",
        f"fast_memory_capacity={problem['fast_memory_capacity']}",
        f"native_granularity={problem['native_granularity']}",
        f"num_ops={len(problem['op_types'])}",
        "Operations:",
    ]
    for i, op_type in enumerate(problem["op_types"]):
        lines.append(f"  Op[{i}]: {op_type}  inputs={problem['inputs'][i]}  outputs={problem['outputs'][i]}")
    return "\n".join(lines)
