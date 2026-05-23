"""
Parallel Strategy Planner.
Runs THREE concurrent LLM calls, each with a different scheduling strategy:
  - AGGRESSIVE: maximize operation fusion (fewest subgraphs)
  - BALANCED: balance compute efficiency and memory transfers
  - CONSERVATIVE: maximize granularity (least splitting), accept more spilling
Each agent receives the same analysis context but different strategy instructions.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from core.dag_utils import compute_analysis, format_analysis


_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "system_planning.txt").read_text()


_STRATEGY_INSTRUCTIONS = {
    "aggressive": """
## Strategy: AGGRESSIVE FUSION
Your goal is to maximize operation grouping to minimize memory transfers.
- Group as many operations as possible into each subgraph
- If needed, use smaller granularity to fit the larger working set
- Fan-out tensors: consider recomputation (run the same op in multiple subgraphs) to avoid retaining large tensors
- Accept sub-native granularity if it enables useful grouping
- Prioritize: fewer subgraphs > larger granularity
""",
    "balanced": """
## Strategy: BALANCED
Your goal is to balance compute efficiency (large granularity) and low memory transfer (grouping).
- Group operations along the critical path
- Use native granularity where possible; accept halved granularity when needed
- Retain tensors at fan-out points only if the memory fits with native granularity
- If MatMul chains exist, use Split-K to group them
- Prioritize: maintain native granularity > extra grouping
""",
    "conservative": """
## Strategy: CONSERVATIVE SPILLING
Your goal is to use the largest possible granularity, accepting more memory transfers.
- Prefer native granularity over grouping
- Allow tensors to spill to slow memory between subgraphs
- Only group operations if they naturally fit at native granularity
- Use tensors_to_retain=[] unless retaining clearly avoids expensive reloads
- Prioritize: native granularity always > grouping
""",
}


class PlannerAgent(BaseAgent):
    """Generates a schedule using one named strategy."""

    async def plan(
        self,
        problem: dict[str, Any],
        analysis_text: str,
        strategy: str,
    ) -> dict[str, Any] | None:
        dag_analysis = compute_analysis(problem)
        dag_text = format_analysis(dag_analysis)
        problem_text = _format_problem(problem)
        strategy_instr = _STRATEGY_INSTRUCTIONS.get(strategy, "")

        prompt = f"""{_SYSTEM_PROMPT}

{dag_text}

{problem_text}

## Expert Analysis (from Analyzer Agent):

{analysis_text}

{strategy_instr}

## Critical Requirements:

1. Every operation must appear in exactly ONE subgraph
2. Working set for each step MUST fit in {problem['fast_memory_capacity']} bytes
3. Subgraphs must respect dependency order (producer step ≤ consumer step)
4. Granularity is [w, h, k] with positive integers; k matters only for MatMul
5. Output ONLY the JSON object – no markdown, no prose

Output this exact JSON structure:
{{
  "subgraphs": [[0, 1], [2]],
  "granularities": [[128, 128, 1], [64, 64, 128]],
  "tensors_to_retain": [[1], []],
  "traversal_orders": [null, [0, 1, 3, 2]],
  "subgraph_latencies": [3276.8, 2048.0]
}}
"""
        # Scale output tokens with problem size.
        # Thinking is disabled (thinking_budget=0) so the full budget goes to JSON output.
        num_ops = len(problem["op_types"])
        max_tokens = max(8192, min(65536, num_ops * 1000))

        print(
            f"  [Planner/{strategy}] Generating schedule (max_tokens={max_tokens}, thinking=off)...",
            file=sys.stderr,
        )
        text = await self._call_llm(
            prompt,
            temperature=0.2,
            max_tokens=max_tokens,
            json_output=True,
            label=f"Planner/{strategy}",
            thinking_budget=0,
        )
        result = self.extract_json(text)
        if result:
            print(f"  [Planner/{strategy}] Done.", file=sys.stderr)
        else:
            head = text[:300] if text else "<empty>"
            tail = text[-200:] if len(text) > 300 else ""
            snippet = f"{head}…[{len(text)} chars total]…{tail}" if tail else (head or "<empty>")
            print(
                f"  [Planner/{strategy}] Failed to parse JSON ({len(text)} chars). Snippet: {snippet!r}",
                file=sys.stderr,
            )
        return result


async def run_parallel_planners(
    client,
    model_name: str,
    problem: dict[str, Any],
    analysis_text: str,
) -> list[tuple[str, dict[str, Any] | None]]:
    """Run all three planning strategies concurrently. Returns [(strategy, solution_or_None)]."""
    agent = PlannerAgent(client, model_name)
    tasks = [
        agent.plan(problem, analysis_text, strategy)
        for strategy in ("aggressive", "balanced", "conservative")
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    strategies = ("aggressive", "balanced", "conservative")
    return [(s, r if isinstance(r, dict) else None) for s, r in zip(strategies, results)]


def _format_problem(problem: dict[str, Any]) -> str:
    lines = [
        "## Problem",
        f"fast_memory_capacity={problem['fast_memory_capacity']}",
        f"slow_memory_bandwidth={problem['slow_memory_bandwidth']}",
        f"native_granularity={problem['native_granularity']}",
        "Operations:",
    ]
    for i, op_type in enumerate(problem["op_types"]):
        inp = problem["inputs"][i]
        out = problem["outputs"][i]
        cost = problem["base_costs"][i]
        lines.append(f"  Op[{i}]: {op_type}  inputs={inp}  outputs={out}  cost={cost}")
    lines.append("Tensors:")
    for i, (ww, hh) in enumerate(zip(problem["widths"], problem["heights"])):
        lines.append(f"  Tensor[{i}]: {ww}×{hh}={ww * hh}")
    return "\n".join(lines)
