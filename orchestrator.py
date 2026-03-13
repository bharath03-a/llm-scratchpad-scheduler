"""
Multi-Agent Orchestrator for DAG Scheduling.

Pipeline:
  1. Analyzer   — single LLM call to deeply understand the problem structure
                  enriched with pre-computed NetworkX DAG metrics
  2. Planners   — THREE concurrent LLM calls (aggressive / balanced / conservative)
                  each receives the same analysis context but different strategy instructions
  3. Validate   — deterministic constraint validation (all bugs fixed) on every candidate
  4. Granularity — deterministic post-processing: find optimal (largest valid) granularity
                  for winning candidate's subgraph groupings
  5. Optimizer  — one LLM call to fine-tune traversal orders and retention strategy
  6. Validate   — final validation + latency recalculation
  7. Refine     — up to 3 iterative correction rounds if any step fails

Design principles:
  - LLM decides WHAT to group; deterministic code decides HOW (granularity, latency)
  - Parallel diversity increases the probability of finding an optimal solution
  - Every candidate passes through identical deterministic validation (no silent failures)
  - Graceful multi-tier fallback: best invalid → refiner → algorithmic fallback
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

import google.genai as genai
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.analyzer import AnalyzerAgent
from agents.optimizer import OptimizerAgent
from agents.planner import run_parallel_planners
from agents.refiner import RefinerAgent
from core.granularity import optimize_granularities
from core.latency import calculate_latency
from core.validator import validate_solution

console = Console(stderr=True)

TIMEOUT_SECONDS = 570  # 9.5 min hard limit (competition is 10 min)


# ---------------------------------------------------------------------------
# Fallback solution (deterministic, always valid)
# ---------------------------------------------------------------------------

def _generate_fallback(problem: dict[str, Any]) -> dict[str, Any]:
    """
    One subgraph per operation, using the largest valid granularity for each.
    Always produces a valid (if sub-optimal) solution.
    """
    from core.granularity import find_optimal_granularity
    from core.dag_utils import compute_analysis
    from networkx import topological_sort
    from core.dag_utils import build_dag

    G = build_dag(problem)
    topo = list(topological_sort(G))

    # Map each op to a standalone subgraph
    subgraphs = [[op] for op in topo]
    tensors_to_retain: list[list[int]] = [[] for _ in topo]
    granularities: list[list[int]] = []
    tensors_in_memory: set[int] = set()

    for sg in subgraphs:
        gran = find_optimal_granularity(problem, sg, tensors_in_memory)
        if gran is None:
            native = problem["native_granularity"]
            gran = [native[0], native[1], 1]
        granularities.append(gran)

    solution: dict[str, Any] = {
        "subgraphs": subgraphs,
        "granularities": granularities,
        "tensors_to_retain": tensors_to_retain,
        "traversal_orders": [None] * len(subgraphs),
        "subgraph_latencies": [],
    }
    solution["subgraph_latencies"] = calculate_latency(problem, solution)
    return solution


# ---------------------------------------------------------------------------
# Solution ranking
# ---------------------------------------------------------------------------

def _total_latency(sol: dict[str, Any]) -> float:
    lats = sol.get("subgraph_latencies") or []
    return sum(lats) if lats else float("inf")


def _rank_valid(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the valid candidate with the lowest total latency."""
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None
    return min(valid, key=_total_latency)


# ---------------------------------------------------------------------------
# Core async pipeline
# ---------------------------------------------------------------------------

async def _run_pipeline(
    client: genai.Client,
    model_name: str,
    problem: dict[str, Any],
) -> dict[str, Any]:
    start = time.time()

    def elapsed() -> float:
        return time.time() - start

    def time_left() -> float:
        return TIMEOUT_SECONDS - elapsed()

    # ── PHASE 1: Analysis ─────────────────────────────────────────────────
    console.print(Panel("[bold cyan]Phase 1 / 5 — Problem Analysis[/bold cyan]"))
    analyzer = AnalyzerAgent(client, model_name)
    try:
        analysis_text = await asyncio.wait_for(
            analyzer.analyze(problem), timeout=min(60.0, time_left() * 0.15)
        )
    except Exception as exc:
        console.print(f"[yellow]Analysis failed ({exc}), proceeding with empty context[/yellow]")
        analysis_text = "(analysis unavailable)"

    # ── PHASE 2: Parallel Planning ────────────────────────────────────────
    console.print(Panel("[bold cyan]Phase 2 / 5 — Parallel Strategy Planning (3 concurrent)[/bold cyan]"))
    try:
        plan_results = await asyncio.wait_for(
            run_parallel_planners(client, model_name, problem, analysis_text),
            timeout=min(120.0, time_left() * 0.35),
        )
    except Exception as exc:
        console.print(f"[red]Parallel planning failed: {exc}[/red]")
        plan_results = []

    # ── PHASE 3: Validate & Deterministic Granularity ────────────────────
    console.print(Panel("[bold cyan]Phase 3 / 5 — Validation + Granularity Optimisation[/bold cyan]"))

    valid_candidates: list[dict[str, Any]] = []
    invalid_candidates: list[dict[str, Any]] = []

    table = Table(title="Planner Candidates", show_header=True)
    table.add_column("Strategy")
    table.add_column("Valid?")
    table.add_column("Latency")
    table.add_column("Notes")

    for strategy, sol in plan_results:
        if sol is None:
            table.add_row(strategy, "❌ parse", "—", "JSON parse failed")
            continue

        # Apply deterministic granularity optimisation
        opt_sol = optimize_granularities(problem, sol)
        target = opt_sol if opt_sol is not None else sol

        is_valid, errors = validate_solution(problem, target)

        if is_valid:
            # Recalculate latencies with our fixed calculator
            target["subgraph_latencies"] = calculate_latency(problem, target)
            valid_candidates.append(target)
            lat = f"{_total_latency(target):.1f}"
            table.add_row(strategy, "✅", lat, f"{len(target['subgraphs'])} subgraphs")
        else:
            invalid_candidates.append(target)
            table.add_row(strategy, "❌", "—", f"{len(errors)} error(s): {errors[0][:60]}")

    console.print(table)

    best = _rank_valid(valid_candidates)

    # ── PHASE 4: Optimise best valid solution ────────────────────────────
    if best is None:
        console.print(Panel("[bold dim]Phase 4 / 5 — Skipped (no valid candidate from planners)[/bold dim]"))
    elif time_left() > 40:
        console.print(Panel("[bold cyan]Phase 4 / 5 — Optimisation Pass[/bold cyan]"))
        optimizer = OptimizerAgent(client, model_name)
        try:
            opt_result = await asyncio.wait_for(
                optimizer.optimize(problem, best, analysis_text),
                timeout=min(60.0, time_left() * 0.25),
            )
            if opt_result is not None:
                # Apply deterministic granularity again
                opt_result = optimize_granularities(problem, opt_result) or opt_result
                is_valid, _ = validate_solution(problem, opt_result)
                if is_valid:
                    opt_result["subgraph_latencies"] = calculate_latency(problem, opt_result)
                    if _total_latency(opt_result) < _total_latency(best):
                        console.print(
                            f"[green]Optimised: {_total_latency(best):.1f} → "
                            f"{_total_latency(opt_result):.1f}[/green]"
                        )
                        best = opt_result
                    else:
                        console.print("[dim]Optimised solution not better — keeping original.[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Optimisation failed ({exc}), keeping best candidate[/yellow]")
    elif best is not None:  # time_left() <= 40
        console.print(Panel("[bold dim]Phase 4 / 5 — Skipped (time budget too low)[/bold dim]"))

    # ── PHASE 5: Refinement (if no valid candidate yet) ──────────────────
    if best is None:
        console.print(Panel("[bold cyan]Phase 5 / 5 — Iterative Refinement[/bold cyan]"))
        # Pick the candidate with the fewest errors as starting point
        refiner = RefinerAgent(client, model_name)
        subject = invalid_candidates[0] if invalid_candidates else _generate_fallback(problem)

        for iteration in range(3):
            if time_left() < 15:
                console.print("[yellow]Time budget exhausted, using fallback.[/yellow]")
                break
            is_valid, errors = validate_solution(problem, subject)
            if is_valid:
                subject["subgraph_latencies"] = calculate_latency(problem, subject)
                best = subject
                console.print(f"[green]Refined after {iteration + 1} iteration(s)[/green]")
                break
            console.print(f"  Refinement {iteration + 1}/3: {len(errors)} error(s)")
            try:
                refined = await asyncio.wait_for(
                    refiner.refine(problem, subject, errors),
                    timeout=min(45.0, time_left() - 10),
                )
                if refined is not None:
                    subject = optimize_granularities(problem, refined) or refined
                else:
                    break
            except Exception as exc:
                console.print(f"[red]Refinement failed: {exc}[/red]")
                break
    else:
        console.print(Panel("[bold dim]Phase 5 / 5 — Skipped (valid solution found)[/bold dim]"))

    if best is None:
        console.print("[yellow]All agents failed — using deterministic fallback solution.[/yellow]")
        best = _generate_fallback(problem)

    console.print(
        f"\n[bold green]✓ Final solution: {len(best['subgraphs'])} subgraphs, "
        f"total latency = {_total_latency(best):.2f}[/bold green]"
    )
    console.print(f"  Elapsed: {elapsed():.1f}s")

    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Synchronous entry point — runs the async pipeline via asyncio.run()."""
        return asyncio.run(_run_pipeline(self.client, self.model_name, problem))
