"""
Fixed latency calculation.
BUG-5 fix: use the external outputs of a subgraph (not just first op's output) for tile counts.
Handles split-K multi-step accumulation for MatMul operations.
"""
from __future__ import annotations

import math
from typing import Any


def _external_outputs(subgraph: list[int], problem: dict[str, Any]) -> set[int]:
    produced: set[int] = set()
    for op in subgraph:
        produced.update(problem["outputs"][op])
    consumed_internally: set[int] = set()
    for op in subgraph:
        for t in problem["inputs"][op]:
            if t in produced:
                consumed_internally.add(t)
    return produced - consumed_internally


def _subgraph_has_matmul(subgraph: list[int], problem: dict[str, Any]) -> bool:
    return any(problem["op_types"][op] == "MatMul" for op in subgraph)


def _k_dimension(subgraph: list[int], problem: dict[str, Any]) -> int:
    """Return the K-reduction dimension for the first MatMul in the subgraph."""
    for op in subgraph:
        if problem["op_types"][op] == "MatMul":
            inp = problem["inputs"][op]
            if inp:
                lhs = inp[0]
                return problem["widths"][lhs]  # width of LHS = K
    return 1


def calculate_subgraph_latency(
    problem: dict[str, Any],
    subgraph: list[int],
    granularity: list[int],
    tensors_retained_before: set[int],
    tensors_to_retain_after: set[int] | None = None,
) -> float:
    """
    Compute the total latency for executing *subgraph* with *granularity*.

    Model (from problem spec):
      - tile_latency = max(compute_per_tile, memory_per_tile)
      - total_latency = sum over tiles  (first / last tiles may differ for split-K)

    For split-K MatMul:
      - First tile: load LHS fully + first RHS strip + write accumulator
      - Middle tiles: load next RHS strip (LHS + accumulator stay resident)
      - Last tile: load next RHS strip + evict final output
    For Pointwise or single-tile:
      - All tiles are symmetric: load slice + write output
    """
    if not subgraph:
        return 0.0

    w, h, k = granularity
    bandwidth = problem["slow_memory_bandwidth"]
    native_w, native_h = problem["native_granularity"]

    ext_outs = _external_outputs(subgraph, problem)
    # Retained outputs are NOT evicted to slow memory → no output transfer cost for them
    evicted_outs = ext_outs - (tensors_to_retain_after or set())

    # --- Tile count ---
    # Use the external output tensors to determine spatial tile grid
    if ext_outs:
        rep_tensor = min(ext_outs)
    else:
        outs_first = problem["outputs"][subgraph[0]]
        rep_tensor = outs_first[0] if outs_first else -1

    if rep_tensor >= 0:
        out_w = problem["widths"][rep_tensor]
        out_h = problem["heights"][rep_tensor]
    else:
        out_w, out_h = native_w, native_h

    tiles_w = math.ceil(out_w / w)
    tiles_h = math.ceil(out_h / h)
    num_spatial = tiles_w * tiles_h

    has_mm = _subgraph_has_matmul(subgraph, problem)
    if has_mm:
        k_dim = _k_dimension(subgraph, problem)
        num_k_splits = math.ceil(k_dim / k)
    else:
        num_k_splits = 1

    # --- Compute cost per tile ---
    # base_cost is paid once per tile regardless of granularity (hardware pads to native)
    compute_per_tile = sum(problem["base_costs"][op] for op in subgraph)

    # --- Identify external inputs (not produced within subgraph, not already retained) ---
    produced_here: set[int] = set()
    for op in subgraph:
        produced_here.update(problem["outputs"][op])

    external_inputs: list[tuple[int, int]] = []  # (tensor_idx, consuming_op)
    seen: set[int] = set()
    for op in subgraph:
        for t in problem["inputs"][op]:
            if t in produced_here or t in tensors_retained_before or t in seen:
                continue
            seen.add(t)
            external_inputs.append((t, op))

    # --- Memory per tile (flat model for Pointwise / simple MatMul) ---
    if not has_mm or num_k_splits == 1:
        # Symmetric: each tile loads its input slices + writes output
        input_mem = 0.0
        for t, op in external_inputs:
            op_type = problem["op_types"][op]
            op_inputs = problem["inputs"][op]
            if op_type == "MatMul" and len(op_inputs) >= 2:
                if t == op_inputs[0]:
                    input_mem += k * h
                else:
                    input_mem += w * k
            else:
                input_mem += w * h
        output_mem = len(evicted_outs) * w * h
        mem_per_tile = (input_mem + output_mem) / bandwidth
        # For MatMul with k_splits, divide compute proportionally per split
        effective_compute = compute_per_tile / max(num_k_splits, 1) if has_mm else compute_per_tile
        tile_lat = max(effective_compute, mem_per_tile)
        return num_spatial * num_k_splits * tile_lat

    # --- Split-K: asymmetric first / middle / last steps ---
    # For each spatial tile, we do num_k_splits passes.
    # Pass 0: load LHS (full h×k_dim strip) + first RHS strip + (no output yet)
    # Passes 1..n-2: load next RHS strip (LHS + accumulator are resident)
    # Pass n-1: load last RHS strip + evict output

    # Identify LHS and RHS tensors for the FIRST MatMul in the subgraph
    lhs_tensor: int | None = None
    rhs_tensors: list[int] = []
    for t, op in external_inputs:
        op_type = problem["op_types"][op]
        op_inputs = problem["inputs"][op]
        if op_type == "MatMul" and len(op_inputs) >= 2:
            if t == op_inputs[0]:
                lhs_tensor = t
            else:
                rhs_tensors.append(t)
        # Pointwise external inputs are streamed per-tile (w*h each)

    lhs_size = (problem["widths"][lhs_tensor] * problem["heights"][lhs_tensor]
                if lhs_tensor is not None else 0)
    rhs_strip_size = sum(w * k for t in rhs_tensors)
    pointwise_input_size = sum(w * h for t, op in external_inputs
                               if problem["op_types"][op] != "MatMul")
    output_size = len(evicted_outs) * w * h

    pass0_mem = (lhs_size + rhs_strip_size + pointwise_input_size) / bandwidth
    mid_mem = (rhs_strip_size + pointwise_input_size) / bandwidth
    last_mem = (rhs_strip_size + pointwise_input_size + output_size) / bandwidth

    # For split-K, each pass does k/K_full fraction of the work → divide evenly
    compute_per_split = compute_per_tile / num_k_splits

    lat_pass0 = max(compute_per_split, pass0_mem)
    lat_mid = max(compute_per_split, mid_mem)
    lat_last = max(compute_per_split, last_mem)

    if num_k_splits == 1:
        per_spatial = lat_last
    elif num_k_splits == 2:
        per_spatial = lat_pass0 + lat_last
    else:
        per_spatial = lat_pass0 + (num_k_splits - 2) * lat_mid + lat_last

    return num_spatial * per_spatial


def calculate_latency(problem: dict[str, Any], solution: dict[str, Any]) -> list[float]:
    """Calculate latency for every subgraph in the solution."""
    subgraphs = solution["subgraphs"]
    granularities = solution["granularities"]
    tensors_to_retain = solution.get("tensors_to_retain", [[] for _ in subgraphs])

    latencies: list[float] = []
    retained: set[int] = set()

    for sg, gran, retain_list in zip(subgraphs, granularities, tensors_to_retain):
        retain_set = set(retain_list)
        lat = calculate_subgraph_latency(problem, sg, gran, retained, retain_set)
        latencies.append(lat)

        # Update retained set for next step
        produced: set[int] = set()
        for op in sg:
            produced.update(problem["outputs"][op])
        retained = {t for t in produced if t in retain_set}

    return latencies
