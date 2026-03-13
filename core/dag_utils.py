"""DAG utilities using NetworkX for structural analysis of scheduling problems."""
from __future__ import annotations

import math
from typing import Any

import networkx as nx


def build_dag(problem: dict[str, Any]) -> nx.DiGraph:
    """Build a directed acyclic graph from the problem definition."""
    G = nx.DiGraph()
    num_ops = len(problem["op_types"])
    G.add_nodes_from(range(num_ops))

    tensor_to_producer: dict[int, int] = {}
    for op_idx, outputs in enumerate(problem["outputs"]):
        for tensor in outputs:
            tensor_to_producer[tensor] = op_idx

    for op_idx, inputs in enumerate(problem["inputs"]):
        for tensor in inputs:
            if tensor in tensor_to_producer:
                producer = tensor_to_producer[tensor]
                if producer != op_idx:
                    G.add_edge(producer, op_idx, tensor=tensor)

    return G


def topological_order(G: nx.DiGraph) -> list[int]:
    return list(nx.topological_sort(G))


def critical_path(G: nx.DiGraph) -> list[int]:
    try:
        return list(nx.dag_longest_path(G))
    except Exception:
        return list(G.nodes())


def fan_out_ops(G: nx.DiGraph) -> list[int]:
    """Operations whose output is consumed by more than one downstream op."""
    return [n for n in G.nodes() if G.out_degree(n) > 1]


def fan_in_ops(G: nx.DiGraph) -> list[int]:
    """Operations that consume outputs from more than one upstream op."""
    return [n for n in G.nodes() if G.in_degree(n) > 1]


def matmul_chains(problem: dict[str, Any], G: nx.DiGraph) -> list[list[int]]:
    """Find chains of consecutive MatMul operations (candidates for Split-K grouping)."""
    chains: list[list[int]] = []
    visited: set[int] = set()

    for node in nx.topological_sort(G):
        if node in visited or problem["op_types"][node] != "MatMul":
            continue
        chain = [node]
        visited.add(node)
        current = node
        while True:
            # Only extend chain if there is a single MatMul successor with no other predecessors
            succs = [
                s for s in G.successors(current)
                if problem["op_types"][s] == "MatMul"
                and s not in visited
                and G.in_degree(s) == 1
            ]
            if not succs:
                break
            nxt = succs[0]
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        if len(chain) > 1:
            chains.append(chain)

    return chains


def level_groups(G: nx.DiGraph) -> list[list[int]]:
    """Assign each op to its BFS level; ops at the same level have no mutual dependency."""
    levels: dict[int, int] = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        levels[node] = max((levels[p] for p in preds), default=-1) + 1

    level_map: dict[int, list[int]] = {}
    for node, lvl in levels.items():
        level_map.setdefault(lvl, []).append(node)

    return [level_map[lvl] for lvl in sorted(level_map)]


def compute_analysis(problem: dict[str, Any]) -> dict[str, Any]:
    """Compute all DAG metrics and return a structured dict."""
    G = build_dag(problem)
    topo = topological_order(G)
    cp = critical_path(G)
    fo = fan_out_ops(G)
    fi = fan_in_ops(G)
    mc = matmul_chains(problem, G)
    lg = level_groups(G)

    capacity = problem["fast_memory_capacity"]
    sizes = [(i, problem["widths"][i] * problem["heights"][i]) for i in range(len(problem["widths"]))]
    sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)
    total_mem = sum(s for _, s in sizes)
    max_tensor = sizes_sorted[0][1] if sizes_sorted else 0

    if max_tensor * 3 <= capacity:
        pressure = "low"
    elif max_tensor * 2 <= capacity:
        pressure = "medium"
    elif max_tensor <= capacity:
        pressure = "high"
    else:
        pressure = "critical"

    op_counts: dict[str, int] = {}
    for ot in problem["op_types"]:
        op_counts[ot] = op_counts.get(ot, 0) + 1

    # Suggest groupings: ops connected in a linear chain are good candidates
    suggested: list[list[int]] = []
    chain_nodes: set[int] = set()
    for node in topo:
        if node in chain_nodes:
            continue
        # Follow linear chains (single successor, single predecessor)
        chain = [node]
        chain_nodes.add(node)
        current = node
        while True:
            succs = list(G.successors(current))
            if len(succs) == 1:
                nxt = succs[0]
                if G.in_degree(nxt) == 1 and nxt not in chain_nodes:
                    chain.append(nxt)
                    chain_nodes.add(nxt)
                    current = nxt
                    continue
            break
        if len(chain) > 1:
            suggested.append(chain)

    return {
        "num_ops": len(problem["op_types"]),
        "num_tensors": len(problem["widths"]),
        "op_type_counts": op_counts,
        "topological_order": topo,
        "critical_path": cp,
        "critical_path_length": len(cp),
        "fan_out_ops": fo,
        "fan_in_ops": fi,
        "matmul_chains": mc,
        "level_groups": lg,
        "total_tensor_memory": total_mem,
        "fast_memory_capacity": capacity,
        "memory_pressure": pressure,
        "largest_tensors": sizes_sorted[:5],
        "suggested_linear_chains": suggested,
    }


def format_analysis(analysis: dict[str, Any]) -> str:
    """Format analysis as rich human-readable text for LLM prompts."""
    lines = [
        "## Computed DAG Analysis",
        f"- Operations: {analysis['num_ops']}  Types: {analysis['op_type_counts']}",
        f"- Tensors: {analysis['num_tensors']}",
        f"- Topological Order: {analysis['topological_order']}",
        f"- Critical Path (longest dependency chain): {analysis['critical_path']}  "
        f"(length {analysis['critical_path_length']})",
        f"- Fan-out ops (output reused by multiple consumers): {analysis['fan_out_ops']}",
        f"- Fan-in ops (depend on multiple producers): {analysis['fan_in_ops']}",
        f"- MatMul Chains (Split-K candidates): {analysis['matmul_chains']}",
        f"- Level Groups (parallel-safe sets): {analysis['level_groups']}",
        f"- Suggested linear fusion chains: {analysis['suggested_linear_chains']}",
        "",
        "## Memory Situation",
        f"- Fast memory capacity: {analysis['fast_memory_capacity']} bytes",
        f"- Total tensor memory: {analysis['total_tensor_memory']} bytes  "
        f"(Pressure: {analysis['memory_pressure'].upper()})",
        f"- Largest tensors [(idx,size)]: {analysis['largest_tensors'][:3]}",
    ]
    return "\n".join(lines)
