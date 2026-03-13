# MLSys 2026 Track B — DAG Scheduling Agent

An agentic solution to the MLSys 2026 Track B scheduling challenge: given a DAG of tensor operations and a hardware memory hierarchy, produce an execution schedule that **minimises total latency** while respecting all fast-memory capacity constraints.

---

## Problem Summary

The hardware has:

- **Slow memory** — infinite capacity, limited bandwidth (all inputs start here, all outputs must end here)
- **Fast memory** — finite scratchpad (e.g. 50 KB), zero-cost access, but strictly capacity-limited

Your scheduler must partition the operation DAG into ordered **subgraphs**, each with an execution granularity `[w, h, k]`, such that the working set of every subgraph fits in fast memory and total latency is minimised.

Full problem specification: [PROBLEM.md](PROBLEM.md)

---

## Architecture

```mermaid
flowchart TD
    INPUT([input.json]) --> MAIN[agent.py\nEntry Point]
    MAIN --> ORCH[Orchestrator\norchestrator.py]

    ORCH --> P1

    subgraph P1["Phase 1 — Analysis (1 LLM call)"]
        DAG[dag_utils.py\nNetworkX pre-compute\ncritical path · fan-out/in\nMatMul chains · levels] --> ANA[AnalyzerAgent\nagents/analyzer.py\nDeep structural analysis\nGrouping · memory · traversal hints]
    end

    P1 --> P2

    subgraph P2["Phase 2 — Parallel Planning (3 concurrent LLM calls)"]
        AGG[PlannerAgent\naggressive\nmax fusion · fewest subgraphs]
        BAL[PlannerAgent\nbalanced\nnative granularity · critical path]
        CON[PlannerAgent\nconservative\nmax granularity · accept spilling]
    end

    P2 --> P3

    subgraph P3["Phase 3 — Deterministic Validation + Granularity (no LLM)"]
        VAL[validator.py\n✓ dependency ordering\n✓ memory constraints\n✓ op coverage + structure]
        GRAN[granularity.py\nFind LARGEST valid w,h,k\nper subgraph algorithmically]
        RANK[Best-of-N Selector\nlowest total latency wins]
        VAL --> GRAN --> RANK
    end

    P3 -->|valid found| P4
    P3 -->|none valid| P5

    subgraph P4["Phase 4 — Optimisation (1 LLM call)"]
        OPT[OptimizerAgent\nagents/optimizer.py\nTraversal orders · snake patterns\nRetention fine-tuning · merge hints]
        REVALIDATE[Re-validate + recalculate\nlatency.py\nsplit-K asymmetric model]
        OPT --> REVALIDATE
    end

    subgraph P5["Phase 5 — Iterative Refinement (up to 3 LLM calls)"]
        REF[RefinerAgent\nagents/refiner.py\nTargeted error correction\nspecific error list per iteration]
        FALLBACK[Deterministic Fallback\none-op-per-subgraph\nalgorithmic granularity]
        REF -->|still invalid| FALLBACK
    end

    P4 --> OUT
    P5 --> OUT
    OUT([output.json])

    subgraph CORE["core/ — Pure Python, no LLM"]
        V2[validator.py]
        L2[latency.py]
        G2[granularity.py]
        D2[dag_utils.py]
    end
```

### Design principles

| Layer                 | Role                                                            | LLM?    |
| --------------------- | --------------------------------------------------------------- | ------- |
| `core/dag_utils.py`   | NetworkX DAG metrics (critical path, fan-out/in, MatMul chains) | No      |
| `core/validator.py`   | Full constraint validation (dependency order, memory, coverage) | No      |
| `core/granularity.py` | Algorithmic search for the largest valid `[w, h, k]`            | No      |
| `core/latency.py`     | Accurate latency model incl. split-K asymmetric passes          | No      |
| `agents/analyzer.py`  | Deep structural analysis enriched with pre-computed metrics     | Yes     |
| `agents/planner.py`   | Three parallel planners with distinct strategies                | Yes × 3 |
| `agents/optimizer.py` | Traversal order + retention fine-tuning                         | Yes     |
| `agents/refiner.py`   | Targeted error correction from specific validation errors       | Yes     |
| `orchestrator.py`     | Async pipeline coordinator, best-of-N selection, fallback       | —       |

**Key insight:** the LLM decides _what_ to group; deterministic Python decides _how_ (optimal granularity, exact latency, constraint satisfaction). This prevents the LLM from producing memory-invalid schedules and removes the need for granularity guessing.

---

## Project Structure

```
.
├── agent.py              # Entry point (CLI: input.json → output.json)
├── orchestrator.py       # Multi-agent pipeline coordinator
├── core/
│   ├── dag_utils.py      # NetworkX DAG analysis
│   ├── validator.py      # Deterministic constraint validation
│   ├── latency.py        # Latency calculation (split-K aware)
│   └── granularity.py    # Optimal granularity search
├── agents/
│   ├── base.py           # Async Gemini base agent + JSON extraction
│   ├── analyzer.py       # Phase 1: problem analysis
│   ├── planner.py        # Phase 2: parallel strategy planning
│   ├── optimizer.py      # Phase 4: schedule optimisation
│   └── refiner.py        # Phase 5: error-correcting refinement
├── prompts/
│   ├── system_analysis.txt
│   ├── system_planning.txt
│   ├── system_optimization.txt
│   ├── system_refinement.txt
│   └── few_shot_examples.json
├── PROBLEM.md            # Full problem specification
├── OPTIMIZATIONS.md      # Implementation notes
└── pyproject.toml
```

---

## Setup

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- A [Google AI Studio](https://aistudio.google.com/) API key with access to Gemini 2.5 Flash

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd llm-scratchpad-scheduler

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Set your API key

```bash
# Option A: environment variable
export GOOGLE_API_KEY=your_key_here

# Option B: .env file in the project root
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### 3. Run

```bash
uv run python3 agent.py input.json output.json
```

The agent writes the solution to `output.json` and streams progress to stderr.

---

## Input / Output Format

### Input (`input.json`)

```json
{
  "widths": [128, 128, 128],
  "heights": [128, 128, 128],
  "inputs": [[0], [1]],
  "outputs": [[1], [2]],
  "base_costs": [1000, 100],
  "op_types": ["Pointwise", "Pointwise"],
  "fast_memory_capacity": 35000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
}
```

### Output (`output.json`)

```json
{
  "subgraphs": [[0, 1]],
  "granularities": [[128, 128, 1]],
  "tensors_to_retain": [[]],
  "traversal_orders": [null],
  "subgraph_latencies": [3276.8]
}
```

See [PROBLEM.md](PROBLEM.md) for the full field specification and worked examples.

---

## Model

The default model is `gemini-2.5-flash`. To use a different model, edit the `model_name` in `orchestrator.py`:

```python
orchestrator = Orchestrator(api_key, model_name="gemini-2.5-pro")
```

---

## Timeout

The agent runs a hard 9.5-minute budget (competition limit is 10 minutes). Phases are skipped if the remaining budget is insufficient — the fallback deterministic solution is always returned within the time limit.

---

## Benchmark Results

Run `uv run python3 score.py` to evaluate all problems in `data/` against the deterministic fallback baseline.

![Benchmark Results](img/image.png)
