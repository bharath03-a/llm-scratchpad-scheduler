"""
Deterministic granularity search.
Given a set of subgraph groupings (decided by the LLM), finds the OPTIMAL (largest)
granularity for each subgraph that still satisfies memory constraints.
This removes the burden from the LLM and guarantees constraint satisfaction.
"""
from __future__ import annotations

import math
from typing import Any

from core.validator import check_memory_constraints


def _candidate_granularities(native_w: int, native_h: int, has_matmul: bool) -> list[list[int]]:
    """
    Generate candidate [w, h, k] granularities in decreasing spatial area order.
    k candidates only matter for MatMul (use k=1 for Pointwise).
    """
    candidates: list[list[int]] = []

    # Generate power-of-2 w and h down to 1
    w_vals = [native_w // (2 ** i) for i in range(int(math.log2(native_w)) + 1) if native_w // (2 ** i) >= 1]
    h_vals = [native_h // (2 ** i) for i in range(int(math.log2(native_h)) + 1) if native_h // (2 ** i) >= 1]

    if has_matmul:
        k_vals = sorted({native_w, native_w // 2, native_w // 4, native_w // 8,
                         native_h, native_h // 2, 64, 32, 16, 8, 4, 1}, reverse=True)
        k_vals = [k for k in k_vals if k >= 1]
    else:
        k_vals = [1]

    for w in w_vals:
        for h in h_vals:
            for k in k_vals:
                candidates.append([w, h, k])

    # Sort by spatial area descending (prefer large tiles), break ties by k descending
    candidates.sort(key=lambda c: (c[0] * c[1], c[2]), reverse=True)
    return candidates


def find_optimal_granularity(
    problem: dict[str, Any],
    subgraph: list[int],
    tensors_in_memory: set[int],
) -> list[int] | None:
    """
    Return the largest [w, h, k] granularity for *subgraph* that fits in fast memory,
    or None if no valid granularity exists.
    """
    native_w, native_h = problem["native_granularity"]
    has_matmul = any(problem["op_types"][op] == "MatMul" for op in subgraph)

    for gran in _candidate_granularities(native_w, native_h, has_matmul):
        ok, _, _ = check_memory_constraints(problem, subgraph, gran, tensors_in_memory)
        if ok:
            return gran

    return None  # No valid granularity – subgraph grouping is infeasible


def optimize_granularities(
    problem: dict[str, Any],
    solution: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Post-process a solution by replacing LLM-chosen granularities with
    the deterministically optimal (largest valid) ones.
    Returns updated solution dict or None if any subgraph has no valid granularity.
    """
    subgraphs = solution["subgraphs"]
    tensors_to_retain = solution.get("tensors_to_retain", [[] for _ in subgraphs])

    new_granularities: list[list[int]] = []
    tensors_in_memory: set[int] = set()

    for sg, retain_list in zip(subgraphs, tensors_to_retain):
        gran = find_optimal_granularity(problem, sg, tensors_in_memory)
        if gran is None:
            return None  # This grouping is infeasible
        new_granularities.append(gran)

        # Update memory state
        produced: set[int] = set()
        for op in sg:
            produced.update(problem["outputs"][op])
        tensors_in_memory = {t for t in produced if t in set(retain_list)}

    return {**solution, "granularities": new_granularities}
