#!/usr/bin/env python3
"""
MLSys 2026 Track B: Agent Reasoning Solution
Uses Google Gemini API to solve DAG scheduling problems with memory constraints.

Optimized with:
- Structured output for all LLM calls
- Iterative refinement with specific error feedback
- Memory-aware validation before LLM calls
- Enhanced prompts with chain-of-thought reasoning
- Better error recovery strategies
"""

import json
import os
import sys
import time
import re
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dotenv import load_dotenv
import google.genai as genai


# ============================================================================
# Helper Functions
# ============================================================================

def build_dag_structure(problem: Dict[str, Any]) -> Dict[int, Set[int]]:
    """Build adjacency list representation of the DAG."""
    num_ops = len(problem['op_types'])
    dag = {i: set() for i in range(num_ops)}
    
    # Map tensor to producing operation
    tensor_to_producer = {}
    for op_idx, outputs in enumerate(problem['outputs']):
        for tensor in outputs:
            tensor_to_producer[tensor] = op_idx
    
    # Build edges: op_i -> op_j if op_j consumes output of op_i
    for op_idx, inputs in enumerate(problem['inputs']):
        for tensor in inputs:
            if tensor in tensor_to_producer:
                producer = tensor_to_producer[tensor]
                dag[producer].add(op_idx)
    
    return dag


def check_memory_constraints(
    problem: Dict[str, Any],
    subgraph: List[int],
    granularity: List[int],
    tensors_in_memory: Set[int]
) -> Tuple[bool, str, int]:
    """
    Check if a subgraph execution fits in memory.
    Returns (is_valid, error_message, working_set_size).
    """
    w, h, k = granularity
    fast_memory_capacity = problem['fast_memory_capacity']
    
    # Calculate working set size
    working_set = 0
    
    # Add tensors already in memory
    for tensor_idx in tensors_in_memory:
        width = problem['widths'][tensor_idx]
        height = problem['heights'][tensor_idx]
        working_set += width * height
    
    # Add input tensors needed for this subgraph
    input_tensors = set()
    for op_idx in subgraph:
        for tensor_idx in problem['inputs'][op_idx]:
            if tensor_idx not in tensors_in_memory:
                input_tensors.add(tensor_idx)
    
    for tensor_idx in input_tensors:
        op_idx = subgraph[0]  # Use first op to determine slice size
        op_type = problem['op_types'][op_idx]
        
        width = problem['widths'][tensor_idx]
        height = problem['heights'][tensor_idx]
        
        if op_type == "MatMul":
            # For MatMul, inputs are sliced differently
            # LHS: k x h, RHS: w x k
            inputs = problem['inputs'][op_idx]
            if tensor_idx == inputs[0]:  # LHS
                slice_size = k * h
            else:  # RHS
                slice_size = w * k
        else:  # Pointwise
            slice_size = w * h
        
        working_set += slice_size
    
    # Add output tensors
    output_tensors = set()
    for op_idx in subgraph:
        for tensor_idx in problem['outputs'][op_idx]:
            output_tensors.add(tensor_idx)
    
    for tensor_idx in output_tensors:
        slice_size = w * h
        working_set += slice_size
    
    if working_set > fast_memory_capacity:
        return False, f"Working set {working_set} exceeds capacity {fast_memory_capacity}", working_set
    
    return True, "", working_set


def validate_solution(problem: Dict[str, Any], solution: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that a solution covers all operations and has correct structure.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    if not isinstance(solution, dict):
        return False, ["Solution must be a dictionary"]
    
    required_keys = ["subgraphs", "granularities", "tensors_to_retain"]
    for key in required_keys:
        if key not in solution:
            errors.append(f"Missing required key: {key}")
            return False, errors
        if not isinstance(solution[key], list):
            errors.append(f"{key} must be a list")
            return False, errors
    
    subgraphs = solution["subgraphs"]
    granularities = solution["granularities"]
    tensors_to_retain = solution["tensors_to_retain"]
    
    # Check lengths match
    if not (len(subgraphs) == len(granularities) == len(tensors_to_retain)):
        errors.append(
            f"Length mismatch: subgraphs={len(subgraphs)}, "
            f"granularities={len(granularities)}, "
            f"tensors_to_retain={len(tensors_to_retain)}"
        )
        return False, errors
    
    # Check all operations are covered
    num_ops = len(problem['op_types'])
    covered_ops = set()
    for i, subgraph in enumerate(subgraphs):
        if not isinstance(subgraph, list):
            errors.append(f"Subgraph {i} is not a list")
            return False, errors
        for op_idx in subgraph:
            if not isinstance(op_idx, int) or op_idx < 0 or op_idx >= num_ops:
                errors.append(
                    f"Invalid operation index {op_idx} in subgraph {i} "
                    f"(valid range: 0-{num_ops-1})"
                )
                return False, errors
            if op_idx in covered_ops:
                errors.append(f"Operation {op_idx} appears multiple times")
                return False, errors
            covered_ops.add(op_idx)
    
    if len(covered_ops) != num_ops:
        missing = sorted(set(range(num_ops)) - covered_ops)
        errors.append(f"Missing operations: {missing}")
        return False, errors
    
    # Check granularities format
    for i, gran in enumerate(granularities):
        if not isinstance(gran, list) or len(gran) != 3:
            errors.append(f"Granularity {i} must be [w, h, k], got {gran}")
            return False, errors
        w, h, k = gran
        if not all(isinstance(x, int) and x > 0 for x in [w, h, k]):
            errors.append(f"Granularity {i} elements must be positive integers, got [{w}, {h}, {k}]")
            return False, errors
    
    # Check memory constraints
    tensors_in_memory: Set[int] = set()
    for step_idx, (subgraph, gran) in enumerate(zip(subgraphs, granularities)):
        is_valid, error_msg, _ = check_memory_constraints(
            problem, subgraph, gran, tensors_in_memory
        )
        if not is_valid:
            errors.append(f"Step {step_idx}: {error_msg}")
            return False, errors
        
        # Update tensors in memory for next step
        for op_idx in subgraph:
            for tensor_idx in problem['outputs'][op_idx]:
                tensors_in_memory.add(tensor_idx)
        
        retain_set = set(tensors_to_retain[step_idx])
        tensors_in_memory = {t for t in tensors_in_memory if t in retain_set}
    
    return True, []


def calculate_latency(problem: Dict[str, Any], solution: Dict[str, Any]) -> List[float]:
    """
    Calculate latency for each subgraph in the solution.
    This is a simplified calculation - the actual evaluation uses more complex logic.
    """
    subgraphs = solution["subgraphs"]
    granularities = solution["granularities"]
    tensors_to_retain = solution.get("tensors_to_retain", [[]] * len(subgraphs))
    traversal_orders = solution.get("traversal_orders", [None] * len(subgraphs))
    
    latencies = []
    slow_bandwidth = problem['slow_memory_bandwidth']
    native_gran = problem['native_granularity']
    
    for i, (subgraph, gran) in enumerate(zip(subgraphs, granularities)):
        w, h, k = gran
        
        # Calculate number of tiles
        if subgraph:
            first_op = subgraph[0]
            output_tensors = problem['outputs'][first_op]
            if output_tensors:
                output_tensor = output_tensors[0]
                output_width = problem['widths'][output_tensor]
                output_height = problem['heights'][output_tensor]
                
                # Calculate spatial tiles
                tiles_w = math.ceil(output_width / w)
                tiles_h = math.ceil(output_height / h)
                num_spatial_tiles = tiles_w * tiles_h
                
                # Calculate K splits for MatMul
                op_type = problem['op_types'][first_op]
                if op_type == "MatMul":
                    inputs = problem['inputs'][first_op]
                    if len(inputs) >= 2:
                        lhs_tensor = inputs[0]
                        k_dim = problem['widths'][lhs_tensor]
                        num_k_splits = math.ceil(k_dim / k)
                    else:
                        num_k_splits = 1
                else:
                    num_k_splits = 1
                
                total_tiles = num_spatial_tiles * num_k_splits
            else:
                total_tiles = 1
        else:
            total_tiles = 1
        
        # Calculate compute time per tile
        compute_per_tile = 0
        for op_idx in subgraph:
            base_cost = problem['base_costs'][op_idx]
            op_type = problem['op_types'][op_idx]
            
            # Adjust for granularity vs native
            native_w, native_h = native_gran
            if w < native_w or h < native_h:
                # Padding overhead
                compute_per_tile += base_cost * (native_w * native_h) / (w * h)
            else:
                compute_per_tile += base_cost
        
        # Calculate memory transfer time per tile
        input_size = 0
        for op_idx in subgraph:
            for tensor_idx in problem['inputs'][op_idx]:
                op_type = problem['op_types'][op_idx]
                
                if op_type == "MatMul":
                    inputs = problem['inputs'][op_idx]
                    if tensor_idx == inputs[0]:  # LHS
                        input_size += k * h
                    else:  # RHS
                        input_size += w * k
                else:  # Pointwise
                    input_size += w * h
        
        output_size = w * h
        memory_per_tile = (input_size + output_size) / slow_bandwidth
        
        # Per-tile latency is max of compute and memory
        tile_latency = max(compute_per_tile, memory_per_tile)
        
        # Total latency
        total_latency = total_tiles * tile_latency
        latencies.append(total_latency)
    
    return latencies


# ============================================================================
# Prompt Manager
# ============================================================================

class PromptManager:
    """Manages prompts for different stages of the agentic reasoning process."""
    
    def __init__(self):
        """Initialize prompt manager and load prompt templates."""
        self.prompts_dir = Path(__file__).parent / "prompts"
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompt templates from files."""
        self.analysis_prompt_template = self._read_prompt("system_analysis.txt")
        self.planning_prompt_template = self._read_prompt("system_planning.txt")
        self.optimization_prompt_template = self._read_prompt("system_optimization.txt")
        self.refinement_prompt_template = self._read_prompt("system_refinement.txt")
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _read_prompt(self, filename: str) -> str:
        """Read a prompt file."""
        filepath = self.prompts_dir / filename
        if filepath.exists():
            return filepath.read_text()
        return f"Please analyze and solve the following scheduling problem.\n"
    
    def _load_few_shot_examples(self) -> Dict[str, str]:
        """Load few-shot examples."""
        examples = {"small": "", "medium": "", "large": ""}
        examples_file = self.prompts_dir / "few_shot_examples.json"
        if examples_file.exists():
            try:
                loaded = json.loads(examples_file.read_text())
                examples.update(loaded)
            except json.JSONDecodeError:
                pass
        return examples
    
    def get_analysis_prompt(self, problem: Dict[str, Any]) -> str:
        """Generate analysis prompt with problem context."""
        problem_str = self._format_problem(problem)
        dag_info = self._format_dag_info(problem)
        
        prompt = f"""{self.analysis_prompt_template}

## Problem to Analyze:

{problem_str}

{dag_info}

## Your Task:

Analyze this scheduling problem step by step:

1. **Graph Structure Analysis**:
   - Identify all dependencies and critical paths
   - Find fan-out points (tensors used by multiple operations)
   - Identify potential grouping opportunities

2. **Memory Constraint Analysis**:
   - Calculate total tensor sizes
   - Identify memory bottlenecks
   - Determine feasible granularity ranges

3. **Operation Characteristics**:
   - Classify MatMul vs Pointwise operations
   - Identify chained MatMuls (candidates for split-K)
   - Note high-cost operations

4. **Strategic Recommendations**:
   - Suggest optimal grouping strategies
   - Recommend granularity choices
   - Propose tensor retention strategy

Provide a structured, detailed analysis.
"""
        return prompt
    
    def get_planning_prompt(self, problem: Dict[str, Any], analysis: str) -> str:
        """Generate planning prompt with analysis context."""
        problem_str = self._format_problem(problem)
        examples = self._get_relevant_examples(problem)
        memory_constraints = self._format_memory_constraints(problem)
        
        prompt = f"""{self.planning_prompt_template}

## Problem:

{problem_str}

{memory_constraints}

## Analysis from Previous Step:

{analysis}

## Examples of Good Solutions:

{examples}

## Critical Requirements:

1. **Coverage**: Every operation must appear in exactly ONE subgraph
2. **Memory Safety**: Working set for each subgraph MUST fit in {problem['fast_memory_capacity']} bytes
3. **Dependencies**: If Op A produces a tensor consumed by Op B, A's subgraph must execute before B's
4. **Granularity**: Use native_granularity {problem['native_granularity']} when possible for efficiency
5. **Format**: Output ONLY valid JSON, no markdown, no explanations

## Your Task:

Generate a complete execution schedule. Think step by step:

1. Group operations that share data (reduce memory transfers)
2. Choose granularities that balance compute efficiency and memory fit
3. Decide which tensors to retain (keep reused tensors in fast memory)
4. Consider traversal orders for data reuse (especially MatMul operations)

Output ONLY the JSON object matching this exact structure:
{{
  "subgraphs": [[0, 1], [2]],
  "granularities": [[128, 128, 1], [64, 64, 128]],
  "tensors_to_retain": [[1], []],
  "traversal_orders": [null, [0, 1, 3, 2]],
  "subgraph_latencies": [3276.8, 2048.0]
}}
"""
        return prompt
    
    def get_optimization_prompt(self, problem: Dict[str, Any], plan: str) -> str:
        """Generate optimization prompt."""
        problem_str = self._format_problem(problem)
        memory_constraints = self._format_memory_constraints(problem)
        
        prompt = f"""{self.optimization_prompt_template}

## Problem:

{problem_str}

{memory_constraints}

## Current Plan:

{plan}

## Optimization Opportunities:

1. **Grouping**: Can operations be better grouped to reduce memory transfers?
2. **Granularity**: Is granularity optimal? Too small wastes compute, too large may not fit
3. **Retention**: Are we retaining the right tensors? Consider recomputation costs
4. **Traversal**: Can traversal order improve data reuse (snake patterns for MatMul)?
5. **Split-K**: For chained MatMuls, can smaller k reduce memory pressure?

## Your Task:

Optimize this schedule to minimize total latency while respecting all constraints.

Output ONLY the optimized JSON schedule (same structure as before).
"""
        return prompt
    
    def get_refinement_prompt(self, problem: Dict[str, Any], solution: Dict[str, Any], errors: List[str]) -> str:
        """Generate refinement prompt with specific error feedback."""
        problem_str = self._format_problem(problem)
        solution_str = self._format_solution(solution)
        memory_constraints = self._format_memory_constraints(problem)
        
        errors_str = "\n".join(f"- {error}" for error in errors)
        
        prompt = f"""{self.refinement_prompt_template}

## Problem:

{problem_str}

{memory_constraints}

## Current Solution (with errors):

{solution_str}

## Specific Errors Found:

{errors_str}

## Your Task:

Fix ALL errors in the solution. Ensure:

1. All operations appear exactly once
2. All lists have matching lengths
3. Granularities are valid [w, h, k] tuples with positive integers
4. Memory constraints are respected (working set ≤ {problem['fast_memory_capacity']})
5. Dependencies are respected (producer executes before consumer)
6. JSON structure is correct

Output ONLY the corrected JSON solution.
"""
        return prompt
    
    def _format_problem(self, problem: Dict[str, Any]) -> str:
        """Format problem as a readable string."""
        lines = [
            f"Fast Memory Capacity: {problem['fast_memory_capacity']} bytes",
            f"Slow Memory Bandwidth: {problem['slow_memory_bandwidth']} bytes/time",
            f"Native Granularity: {problem['native_granularity']}",
            "",
            "Operations:",
        ]
        
        for i, op_type in enumerate(problem['op_types']):
            inputs = problem['inputs'][i]
            outputs = problem['outputs'][i]
            cost = problem['base_costs'][i]
            lines.append(
                f"  Op[{i}]: {op_type}, inputs={inputs}, outputs={outputs}, cost={cost}"
            )
        
        lines.append("")
        lines.append("Tensors:")
        for i, (w, h) in enumerate(zip(problem['widths'], problem['heights'])):
            size = w * h
            lines.append(f"  Tensor[{i}]: {w}x{h} (size={size} bytes)")
        
        return "\n".join(lines)
    
    def _format_dag_info(self, problem: Dict[str, Any]) -> str:
        """Format DAG structure information."""
        dag = build_dag_structure(problem)
        lines = ["", "DAG Structure (dependencies):"]
        
        for op_idx, successors in dag.items():
            if successors:
                lines.append(f"  Op[{op_idx}] -> {sorted(successors)}")
        
        if not any(dag.values()):
            lines.append("  (No dependencies - all operations are independent)")
        
        return "\n".join(lines)
    
    def _format_memory_constraints(self, problem: Dict[str, Any]) -> str:
        """Format memory constraint information."""
        capacity = problem['fast_memory_capacity']
        lines = [
            "",
            "## Memory Constraints:",
            f"- Fast memory capacity: {capacity} bytes",
            f"- Each subgraph's working set (input slices + output slices) must fit in {capacity} bytes",
            f"- Working set calculation:",
            f"  * For Pointwise: input_size = w×h, output_size = w×h",
            f"  * For MatMul: LHS = k×h, RHS = w×k, output = w×h",
            f"  * Plus any tensors retained from previous steps",
        ]
        return "\n".join(lines)
    
    def _format_solution(self, solution: Dict[str, Any]) -> str:
        """Format solution as a readable string."""
        return json.dumps(solution, indent=2)
    
    def _get_relevant_examples(self, problem: Dict[str, Any]) -> str:
        """Get relevant few-shot examples based on problem characteristics."""
        num_ops = len(problem['op_types'])
        
        if num_ops <= 5:
            return self.few_shot_examples.get("small", "")
        elif num_ops <= 20:
            return self.few_shot_examples.get("medium", "")
        else:
            return self.few_shot_examples.get("large", "")


# ============================================================================
# Scheduling Agent
# ============================================================================

class SchedulingAgent:
    """Multi-step agentic scheduler using Gemini API with iterative refinement."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """Initialize the agent with Gemini API."""
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.prompt_manager = PromptManager()
        self.max_retries = 3
        self.max_refinement_iterations = 3
        self.timeout_seconds = 600  # 10 minutes total
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main solving method using multi-step agentic reasoning with iterative refinement.
        
        Steps:
        1. Analyze the problem structure
        2. Plan initial schedule (with structured output)
        3. Optimize the schedule (with structured output)
        4. Validate and iteratively refine if needed
        """
        start_time = time.time()
        
        # Step 1: Analyze problem structure
        print("Step 1: Analyzing problem structure...", file=sys.stderr)
        try:
            analysis = self._analyze_problem(problem)
        except Exception as e:
            print(f"Analysis failed: {e}, proceeding with direct planning", file=sys.stderr)
            analysis = ""
        
        if time.time() - start_time > self.timeout_seconds * 0.6:
            print("Warning: Approaching timeout, skipping optimization", file=sys.stderr)
            return self._plan_and_validate(problem, analysis, skip_optimization=True)
        
        # Step 2: Plan initial schedule
        print("Step 2: Planning initial schedule...", file=sys.stderr)
        try:
            initial_plan = self._plan_schedule(problem, analysis)
        except Exception as e:
            print(f"Planning failed: {e}, using fallback", file=sys.stderr)
            return self._generate_fallback_solution(problem)
        
        if time.time() - start_time > self.timeout_seconds * 0.75:
            print("Warning: Approaching timeout, skipping optimization", file=sys.stderr)
            return self._validate_and_refine(problem, initial_plan)
        
        # Step 3: Optimize schedule
        print("Step 3: Optimizing schedule...", file=sys.stderr)
        try:
            optimized = self._optimize_schedule(problem, initial_plan)
        except Exception as e:
            print(f"Optimization failed: {e}, using initial plan", file=sys.stderr)
            optimized = initial_plan
        
        # Step 4: Validate and iteratively refine
        print("Step 4: Validating and refining solution...", file=sys.stderr)
        solution = self._validate_and_refine(problem, optimized)
        
        elapsed = time.time() - start_time
        print(f"Total time: {elapsed:.2f}s", file=sys.stderr)
        
        return solution
    
    def _plan_and_validate(self, problem: Dict[str, Any], analysis: str, skip_optimization: bool = False) -> Dict[str, Any]:
        """Plan and validate without optimization."""
        plan = self._plan_schedule(problem, analysis)
        return self._validate_and_refine(problem, plan)
    
    def _analyze_problem(self, problem: Dict[str, Any]) -> str:
        """Analyze the problem structure and identify key constraints."""
        prompt = self.prompt_manager.get_analysis_prompt(problem)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                    }
                )
                return response.text
            except Exception as e:
                print(f"Analysis attempt {attempt + 1} failed: {e}", file=sys.stderr)
                if attempt == self.max_retries - 1:
                    raise
        
        return ""
    
    def _plan_schedule(self, problem: Dict[str, Any], analysis: str) -> str:
        """Plan the initial execution schedule with structured JSON output."""
        prompt = self.prompt_manager.get_planning_prompt(problem, analysis)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 4096,
                        "response_mime_type": "application/json",  # Structured output
                    }
                )
                return response.text
            except Exception as e:
                print(f"Planning attempt {attempt + 1} failed: {e}", file=sys.stderr)
                if attempt == self.max_retries - 1:
                    raise
        
        return ""
    
    def _optimize_schedule(self, problem: Dict[str, Any], plan: str) -> str:
        """Optimize the schedule for better performance with structured JSON output."""
        prompt = self.prompt_manager.get_optimization_prompt(problem, plan)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 4096,
                        "response_mime_type": "application/json",  # Structured output
                    }
                )
                return response.text
            except Exception as e:
                print(f"Optimization attempt {attempt + 1} failed: {e}", file=sys.stderr)
                if attempt == self.max_retries - 1:
                    return plan
        
        return plan
    
    def _validate_and_refine(self, problem: Dict[str, Any], solution_text: str) -> Dict[str, Any]:
        """Validate and iteratively refine the solution."""
        # Extract JSON from the response
        solution_json = self._extract_json(solution_text)
        
        if not solution_json:
            print("Warning: Could not extract JSON, generating fallback solution", file=sys.stderr)
            return self._generate_fallback_solution(problem)
        
        # Iterative refinement loop
        for iteration in range(self.max_refinement_iterations):
            is_valid, errors = validate_solution(problem, solution_json)
            
            if is_valid:
                # Calculate latencies if missing
                if "subgraph_latencies" not in solution_json or not solution_json["subgraph_latencies"]:
                    solution_json["subgraph_latencies"] = calculate_latency(problem, solution_json)
                
                if iteration > 0:
                    print(f"Solution validated after {iteration} refinement(s)", file=sys.stderr)
                return solution_json
            
            if iteration < self.max_refinement_iterations - 1:
                print(f"Refinement iteration {iteration + 1}: {len(errors)} error(s) found", file=sys.stderr)
                refined = self._refine_solution(problem, solution_json, errors)
                if refined:
                    solution_json = refined
                else:
                    print("Refinement failed, using current solution", file=sys.stderr)
                    break
            else:
                print(f"Max refinements reached. Errors: {errors}", file=sys.stderr)
        
        # If still invalid after refinements, try fallback
        print("Warning: Solution still invalid after refinements, using fallback", file=sys.stderr)
        return self._generate_fallback_solution(problem)
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response text."""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            sample = text[:500] if len(text) > 500 else text
            print(f"JSON parse error: {e}. Sample: {sample}", file=sys.stderr)
            return None
    
    def _refine_solution(self, problem: Dict[str, Any], solution: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
        """Refine an invalid solution with specific error feedback."""
        prompt = self.prompt_manager.get_refinement_prompt(problem, solution, errors)
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 4096,
                    "response_mime_type": "application/json",  # Structured output
                }
            )
            refined = self._extract_json(response.text)
            if refined:
                return refined
        except Exception as e:
            print(f"Refinement failed: {e}", file=sys.stderr)
        
        return None
    
    def _generate_fallback_solution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a basic valid solution as fallback."""
        num_ops = len(problem["op_types"])
        
        # Simple strategy: one subgraph per operation
        subgraphs = [[i] for i in range(num_ops)]
        
        # Use native granularity for all
        native = problem["native_granularity"]
        granularities = [[native[0], native[1], 1] for _ in range(num_ops)]
        
        # Don't retain any tensors (simple spill strategy)
        tensors_to_retain = [[] for _ in range(num_ops)]
        
        # Default traversal orders
        traversal_orders = [None for _ in range(num_ops)]
        
        # Calculate latencies
        solution = {
            "subgraphs": subgraphs,
            "granularities": granularities,
            "tensors_to_retain": tensors_to_retain,
            "traversal_orders": traversal_orders,
            "subgraph_latencies": []
        }
        
        solution["subgraph_latencies"] = calculate_latency(problem, solution)
        
        return solution


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the agent."""
    load_dotenv()
    
    if len(sys.argv) != 3:
        print("Usage: python3 agent.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set", file=sys.stderr)
        print("Please set GOOGLE_API_KEY in your environment or create a .env file", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(input_path, 'r') as f:
            problem = json.load(f)
    except Exception as e:
        print(f"Error loading problem: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        agent = SchedulingAgent(api_key)
        solution = agent.solve(problem)
        
        with open(output_path, 'w') as f:
            json.dump(solution, f, indent=2)
        
        print(f"Solution written to {output_path}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error solving problem: {e}", file=sys.stderr)
        with open(output_path, 'w') as f:
            json.dump({}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()
