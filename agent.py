#!/usr/bin/env python3
"""
MLSys 2026 Track B: Agent Reasoning Solution
Entry point — delegates to the multi-agent orchestrator.

Architecture overview:
  Phase 1  Analyzer        Deep problem analysis (NetworkX DAG metrics + LLM)
  Phase 2  Planners × 3   Parallel: aggressive / balanced / conservative strategies
  Phase 3  Validation      Deterministic constraint checking (all bugs fixed)
  Phase 4  Optimizer       Traversal + retention fine-tuning on best candidate
  Phase 5  Refiner         Iterative repair if no valid solution found
"""

import json
import os
import sys

from dotenv import load_dotenv

from orchestrator import Orchestrator


def main() -> None:
    load_dotenv()

    if len(sys.argv) != 3:
        print("Usage: python3 agent.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path) as f:
            problem = json.load(f)
    except Exception as exc:
        print(f"Error loading problem: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        orchestrator = Orchestrator(api_key)
        solution = orchestrator.solve(problem)

        with open(output_path, "w") as f:
            json.dump(solution, f, indent=2)

        print(f"Solution written to {output_path}", file=sys.stderr)

    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        with open(output_path, "w") as f:
            json.dump({}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()
