#!/usr/bin/env python3
"""
Quick scoring script — no LLM calls.

For each problem in data/, shows:
  1. Fallback latency  (deterministic one-op-per-subgraph baseline)
  2. Output latency    (if a matching file exists in output/)
  3. Improvement ratio

Usage:
  uv run python3 score.py
  uv run python3 score.py --run    # also run the agent on every problem (slow)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rich.console import Console
from rich.table import Table

# Ensure repo-root imports work even when executed as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.latency import calculate_latency
from core.validator import validate_solution
from orchestrator import _generate_fallback

console = Console()


def _total(sol: dict) -> float:
    return sum(sol.get("subgraph_latencies") or [])


def score_problem(problem_path: str, output_path: str | None) -> dict:
    with open(problem_path) as f:
        problem = json.load(f)
    name = os.path.basename(problem_path)

    # ── Baseline: deterministic fallback ──────────────────────────────────
    fallback = _generate_fallback(problem)
    fallback_lat = _total(fallback)
    fallback_sgs = len(fallback["subgraphs"])

    # ── Existing output (if any) ──────────────────────────────────────────
    agent_lat: float | None = None
    agent_sgs: int | None = None
    agent_valid: bool | None = None
    agent_note: str = "—"

    if output_path and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                sol = json.load(f)
            # Recompute latency from scratch to ensure accuracy
            is_valid, errors = validate_solution(problem, sol)
            if is_valid:
                sol["subgraph_latencies"] = calculate_latency(problem, sol)
                agent_lat = _total(sol)
                agent_sgs = len(sol["subgraphs"])
                agent_valid = True
                agent_note = "valid"
            else:
                agent_valid = False
                agent_note = f"INVALID: {errors[0][:60]}"
        except Exception as exc:
            agent_note = f"error: {exc}"

    improvement = (fallback_lat / agent_lat) if agent_lat else None

    return {
        "name": name,
        "num_ops": len(problem["op_types"]),
        "fast_mem": problem["fast_memory_capacity"],
        "fallback_lat": fallback_lat,
        "fallback_sgs": fallback_sgs,
        "agent_lat": agent_lat,
        "agent_sgs": agent_sgs,
        "agent_valid": agent_valid,
        "agent_note": agent_note,
        "improvement": improvement,
    }


def run_agent(problem_path: str, output_path: str) -> None:
    console.print(f"  Running agent on [cyan]{os.path.basename(problem_path)}[/cyan]...")
    start = time.time()
    result = subprocess.run(
        ["uv", "run", "python3", "agent.py", problem_path, output_path],
        capture_output=False,
    )
    elapsed = time.time() - start
    status = "[green]OK[/green]" if result.returncode == 0 else "[red]FAIL[/red]"
    console.print(f"  {status} Done in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score scheduling solutions")
    parser.add_argument("--run", action="store_true", help="Run agent on all problems first")
    args = parser.parse_args()

    # `score.py` lives in `metrics/`, but `data/` and `output/` live at repo root.
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_files = sorted(glob.glob(str(data_dir / "*.json")))
    if not problem_files:
        console.print("[red]No problem files found in data/[/red]")
        sys.exit(1)

    if args.run:
        console.print("[bold]Running agent on all problems...[/bold]")
        for pf in problem_files:
            stem = os.path.splitext(os.path.basename(pf))[0]
            out = str(output_dir / f"{stem}.json")
            run_agent(pf, out)
        console.print()

    # ── Build results table ───────────────────────────────────────────────
    table = Table(title="Benchmark Results", show_header=True)
    table.add_column("Problem", style="cyan")
    table.add_column("Ops", justify="right")
    table.add_column("FastMem", justify="right")
    table.add_column("Baseline latency", justify="right")
    table.add_column("Agent latency", justify="right")
    table.add_column("Agent SGs", justify="right")
    table.add_column("Speedup", justify="right", style="green")
    table.add_column("Status")

    total_baseline = 0.0
    total_agent = 0.0

    for pf in problem_files:
        stem = os.path.splitext(os.path.basename(pf))[0]
        out = str(output_dir / f"{stem}.json")
        r = score_problem(pf, out if os.path.exists(out) else None)

        agent_lat_str = f"{r['agent_lat']:.0f}" if r["agent_lat"] is not None else "—"
        agent_sgs_str = str(r["agent_sgs"]) if r["agent_sgs"] is not None else "—"

        if r["improvement"] is not None:
            speedup = f"{r['improvement']:.2f}×"
            if r["improvement"] >= 2:
                speedup = f"[bold green]{speedup}[/bold green]"
            elif r["improvement"] < 1:
                speedup = f"[red]{speedup}[/red]"
        else:
            speedup = "—"

        if r["agent_valid"] is None:
            status = "[dim]no output[/dim]"
        elif r["agent_valid"]:
            status = "[green]OK[/green]"
        else:
            status = f"[red]{r['agent_note']}[/red]"

        table.add_row(
            r["name"],
            str(r["num_ops"]),
            f"{r['fast_mem']:,}",
            f"{r['fallback_lat']:.0f}",
            agent_lat_str,
            agent_sgs_str,
            speedup,
            status,
        )

        total_baseline += r["fallback_lat"]
        if r["agent_lat"] is not None:
            total_agent += r["agent_lat"]

    console.print(table)

    if total_agent > 0:
        overall = total_baseline / total_agent
        console.print(
            f"\n[bold]Overall speedup (scored problems): {overall:.2f}×[/bold]  "
            f"({total_baseline:.0f} → {total_agent:.0f})"
        )

    console.print(
        "\n[dim]To run the agent on all problems:[/dim] "
        "[cyan]uv run python3 score.py --run[/cyan]"
    )
    console.print(
        "[dim]To run on a single problem:[/dim] "
        "[cyan]uv run python3 agent.py data/<problem>.json output/<problem>.json[/cyan]"
    )


if __name__ == "__main__":
    main()
