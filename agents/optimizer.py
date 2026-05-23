"""
Schedule Optimizer Agent.
Takes the best valid candidate solution and refines:
  - Traversal orders (snake patterns for MatMul)
  - Tensor retention strategy
  - Granularity hints (the deterministic optimizer handles the actual values)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from core.dag_utils import compute_analysis, format_analysis

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "system_optimization.txt").read_text()


class OptimizerAgent(BaseAgent):

    async def optimize(
        self,
        problem: dict[str, Any],
        solution: dict[str, Any],
        analysis_text: str,
    ) -> dict[str, Any] | None:
        dag_analysis = compute_analysis(problem)
        dag_text = format_analysis(dag_analysis)

        import json

        solution_str = json.dumps(solution, indent=2)

        prompt = f"""{_SYSTEM_PROMPT}

{dag_text}

## Expert Analysis:

{analysis_text}

## Current Valid Solution:

{solution_str}

## Optimization Focus

1. **Traversal Orders**: For MatMul subgraphs, consider snake/zig-zag patterns.
   - MatMul chains identified: {dag_analysis['matmul_chains']}
   - A [0,1,3,2] pattern (top-left → top-right → bottom-right → bottom-left) reuses strips.

2. **Tensor Retention**: Review tensors_to_retain.
   - Fan-out ops {dag_analysis['fan_out_ops']} produce tensors used by multiple steps.
   - Retaining is only beneficial if the tensor is needed in the very next step.

3. **Grouping Adjustments**: Can any adjacent singleton subgraphs be safely merged?
   - Memory capacity: {problem['fast_memory_capacity']} bytes
   - Any merge must keep working set within capacity.

4. **Granularity Adjustments** (optional hints only – a deterministic pass will finalize them):
   - Prefer native granularity {problem['native_granularity']} when possible.
   - For Split-K, consider k values that balance memory and compute.

Output ONLY the optimized JSON (same structure, no markdown).
"""
        num_ops = len(problem["op_types"])
        max_tokens = max(8192, min(65536, num_ops * 1000))
        print(
            f"  [Optimizer] Optimizing solution (max_tokens={max_tokens}, thinking=off)...", file=sys.stderr
        )
        text = await self._call_llm(
            prompt,
            temperature=0.25,
            max_tokens=max_tokens,
            json_output=True,
            label="Optimizer",
            thinking_budget=0,
        )
        result = self.extract_json(text)
        if result:
            print("  [Optimizer] Done.", file=sys.stderr)
        else:
            print("  [Optimizer] Failed to parse JSON, keeping current solution.", file=sys.stderr)
        return result
