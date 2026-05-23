"""
Comprehensive, deterministic constraint validation for DAG schedules.
Fixes all bugs from the original implementation:
  - BUG-1: Missing dependency ordering check (now implemented)
  - BUG-2: Mixed-subgraph op-type routing (always used subgraph[0]; now finds actual consuming op)
  - BUG-5: Output tile count from first op only (now uses external-output detection)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tensor_to_producer(problem: dict[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for op_idx, outputs in enumerate(problem["outputs"]):
        for t in outputs:
            mapping[t] = op_idx
    return mapping


def _find_consuming_op(tensor_idx: int, subgraph: list[int], problem: dict[str, Any]) -> int | None:
    """Return the op in *subgraph* that consumes *tensor_idx*, or None."""
    for op in subgraph:
        if tensor_idx in problem["inputs"][op]:
            return op
    return None


def _external_outputs(subgraph: list[int], problem: dict[str, Any]) -> set[int]:
    """Tensors produced by this subgraph that are NOT consumed within it (true outputs)."""
    produced: set[int] = set()
    for op in subgraph:
        produced.update(problem["outputs"][op])
    consumed_internally: set[int] = set()
    for op in subgraph:
        for t in problem["inputs"][op]:
            if t in produced:
                consumed_internally.add(t)
    return produced - consumed_internally


# ---------------------------------------------------------------------------
# Working-set check  (BUG-2 fixed)
# ---------------------------------------------------------------------------


def check_memory_constraints(
    problem: dict[str, Any],
    subgraph: list[int],
    granularity: list[int],
    tensors_in_memory: set[int],
) -> tuple[bool, str, int]:
    """
    Returns (is_valid, error_message, working_set_bytes).

    FIX: For each input tensor we look up the ACTUAL consuming op in the subgraph
    to decide whether it is a MatMul LHS/RHS or a Pointwise input.
    """
    w, h, k = granularity
    capacity = problem["fast_memory_capacity"]
    working_set = 0

    # Tensors already resident from a previous step
    for t in tensors_in_memory:
        working_set += problem["widths"][t] * problem["heights"][t]

    # External inputs needed by this subgraph (not already resident, not ephemeral)
    produced_here: set[int] = set()
    for op in subgraph:
        produced_here.update(problem["outputs"][op])

    seen_inputs: set[int] = set()
    for op in subgraph:
        for t in problem["inputs"][op]:
            if t in tensors_in_memory or t in produced_here or t in seen_inputs:
                continue
            seen_inputs.add(t)

            # FIX: use THIS op's type, not subgraph[0]
            op_type = problem["op_types"][op]
            op_inputs = problem["inputs"][op]

            if op_type == "MatMul" and len(op_inputs) >= 2:
                if t == op_inputs[0]:  # LHS slice: k × h
                    slice_size = k * h
                else:  # RHS slice: w × k
                    slice_size = w * k
            else:  # Pointwise slice: w × h
                slice_size = w * h

            working_set += slice_size

    # Output slices (w × h each)
    ext_outs = _external_outputs(subgraph, problem)
    working_set += len(ext_outs) * w * h

    if working_set > capacity:
        return False, f"Working set {working_set} exceeds capacity {capacity}", working_set
    return True, "", working_set


# ---------------------------------------------------------------------------
# Dependency-ordering validation  (BUG-1: was completely missing)
# ---------------------------------------------------------------------------


def validate_dependency_ordering(
    problem: dict[str, Any],
    subgraphs: list[list[int]],
) -> tuple[bool, list[str]]:
    """
    Verify that for every edge (producer_op → consumer_op) in the DAG,
    the producer's subgraph step comes BEFORE the consumer's subgraph step.
    Same-step ops are allowed (ephemeral, handled by the hardware).
    """
    errors: list[str] = []

    # Map op → step index
    op_to_step: dict[int, int] = {}
    for step, sg in enumerate(subgraphs):
        for op in sg:
            op_to_step[op] = step

    ttp = _tensor_to_producer(problem)

    for op_idx, inputs in enumerate(problem["inputs"]):
        for t in inputs:
            if t not in ttp:
                continue  # graph input tensor – no producer op
            producer_op = ttp[t]
            if producer_op == op_idx:
                continue  # self-loop (shouldn't happen in a DAG)
            prod_step = op_to_step.get(producer_op)
            cons_step = op_to_step.get(op_idx)
            if prod_step is None or cons_step is None:
                continue
            if prod_step > cons_step:
                errors.append(
                    f"Dependency violated: Op[{op_idx}] (step {cons_step}) "
                    f"needs tensor[{t}] produced by Op[{producer_op}] (step {prod_step})"
                )

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Full solution validator
# ---------------------------------------------------------------------------


def validate_solution(
    problem: dict[str, Any],
    solution: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Complete validation including:
      1. Structure & types
      2. Operation coverage (no duplicates, no missing)
      3. Granularity format
      4. Memory constraints per step  (BUG-2 fixed)
      5. Dependency ordering          (BUG-1 added)
    Returns (is_valid, errors).
    """
    errors: list[str] = []

    # --- 1. Top-level structure ---
    if not isinstance(solution, dict):
        return False, ["Solution must be a dict"]

    for key in ("subgraphs", "granularities", "tensors_to_retain"):
        if key not in solution:
            errors.append(f"Missing required key: {key}")
        elif not isinstance(solution[key], list):
            errors.append(f"'{key}' must be a list")

    if errors:
        return False, errors

    subgraphs = solution["subgraphs"]
    granularities = solution["granularities"]
    tensors_to_retain = solution["tensors_to_retain"]

    if not (len(subgraphs) == len(granularities) == len(tensors_to_retain)):
        errors.append(
            f"Length mismatch: subgraphs={len(subgraphs)}, "
            f"granularities={len(granularities)}, "
            f"tensors_to_retain={len(tensors_to_retain)}"
        )
        return False, errors

    # --- 2. Operation coverage ---
    num_ops = len(problem["op_types"])
    covered: set[int] = set()
    for i, sg in enumerate(subgraphs):
        if not isinstance(sg, list):
            errors.append(f"Subgraph {i} is not a list")
            return False, errors
        for op in sg:
            if not isinstance(op, int) or op < 0 or op >= num_ops:
                errors.append(f"Invalid op index {op} in subgraph {i} (valid: 0–{num_ops - 1})")
                return False, errors
            if op in covered:
                errors.append(f"Op {op} appears multiple times")
                return False, errors
            covered.add(op)

    if len(covered) != num_ops:
        missing = sorted(set(range(num_ops)) - covered)
        errors.append(f"Missing operations: {missing}")
        return False, errors

    # --- 3. Granularity format ---
    for i, gran in enumerate(granularities):
        if not isinstance(gran, list) or len(gran) != 3:
            errors.append(f"Granularity {i} must be [w,h,k], got {gran}")
            return False, errors
        w, h, kk = gran
        if not all(isinstance(x, int) and x > 0 for x in [w, h, kk]):
            errors.append(f"Granularity {i} elements must be positive ints, got {gran}")
            return False, errors

    # --- 4. Memory constraints per step ---
    tensors_in_memory: set[int] = set()
    for step, (sg, gran) in enumerate(zip(subgraphs, granularities)):
        ok, msg, _ = check_memory_constraints(problem, sg, gran, tensors_in_memory)
        if not ok:
            errors.append(f"Step {step} memory: {msg}")
            return False, errors

        # Update memory state
        for op in sg:
            tensors_in_memory.update(problem["outputs"][op])
        retain = set(tensors_to_retain[step])
        tensors_in_memory = {t for t in tensors_in_memory if t in retain}

    # --- 5. Dependency ordering  (was missing entirely) ---
    dep_ok, dep_errors = validate_dependency_ordering(problem, subgraphs)
    if not dep_ok:
        errors.extend(dep_errors)
        return False, errors

    return True, []
