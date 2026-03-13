"""
Problem Analyzer Agent.
Single focused job: deeply understand the problem structure and produce
a rich analysis context that all downstream agents will use.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from core.dag_utils import compute_analysis, format_analysis


_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "system_analysis.txt").read_text()


class AnalyzerAgent(BaseAgent):
    """
    Runs a single analytical LLM call enriched with pre-computed DAG metrics.
    Returns a free-text analysis that downstream agents inject into their prompts.
    """

    async def analyze(self, problem: dict[str, Any]) -> str:
        """Return a rich textual analysis of the scheduling problem."""
        dag_analysis = compute_analysis(problem)
        dag_text = format_analysis(dag_analysis)
        problem_text = _format_problem(problem)

        prompt = f"""{_SYSTEM_PROMPT}

{dag_text}

{problem_text}

## Your Task

Based on the computed DAG metrics above, provide a FOCUSED analysis covering:

1. **Grouping Strategy**: Which operations should be grouped together and why?
   - Reference the 'suggested_linear_chains' and 'matmul_chains' above
   - Note fan-out points where grouping choices diverge

2. **Memory Strategy**: How tight are the memory constraints?
   - Memory pressure is {dag_analysis['memory_pressure'].upper()}
   - Identify which subgraph groupings will require reduced granularity

3. **Granularity Recommendation**: What [w, h, k] values are feasible?
   - Native granularity is {problem['native_granularity']}
   - For MatMul chains (Split-K), what k values are recommended?

4. **Retention Strategy**: Which tensors should be retained between steps?
   - Focus on fan-out tensors: {dag_analysis['fan_out_ops']}

5. **Traversal Optimization**: Which subgraphs benefit from snake/zig-zag traversal?

Be specific and concise. Reference operation indices directly.
"""
        print("  [Analyzer] Running deep problem analysis...", file=sys.stderr)
        text = await self._call_llm_text(prompt, temperature=0.1, max_tokens=2048, label="Analyzer")
        print("  [Analyzer] Done.", file=sys.stderr)
        return text


def _format_problem(problem: dict[str, Any]) -> str:
    lines = [
        "## Problem Specification",
        f"Fast Memory Capacity: {problem['fast_memory_capacity']} bytes",
        f"Slow Memory Bandwidth: {problem['slow_memory_bandwidth']} bytes/time",
        f"Native Granularity: {problem['native_granularity']}",
        "",
        "Operations:",
    ]
    for i, op_type in enumerate(problem["op_types"]):
        inp = problem["inputs"][i]
        out = problem["outputs"][i]
        cost = problem["base_costs"][i]
        lines.append(f"  Op[{i}]: {op_type}  inputs={inp}  outputs={out}  base_cost={cost}")
    lines.append("")
    lines.append("Tensors:")
    for i, (ww, hh) in enumerate(zip(problem["widths"], problem["heights"])):
        lines.append(f"  Tensor[{i}]: {ww}×{hh} = {ww * hh} bytes")
    return "\n".join(lines)
