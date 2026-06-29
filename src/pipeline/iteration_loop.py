"""
Iteration Loop
===============
Pipeline Stage 6: Orchestrates Layer 1 (Prompt Optimizer) and
Layer 2 (Confidence Evaluator) in a feedback loop.

Flow:
  1. Optimize the prompt (Layer 1)
  2. Evaluate confidence (Layer 2)
  3. If score < threshold AND iterations < max: feed missing/extra
     tasks back to Layer 1 and repeat
  4. Return the best result with full iteration history

Hard cap of 3 iterations is always enforced.
"""

from __future__ import annotations

import os
import sys

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipeline.classifier import ComplexityLevel
from src.pipeline.prompt_optimizer import optimize_prompt
from src.pipeline.confidence_evaluator import (
    TaskType,
    detect_task_type,
    evaluate_confidence,
)


# ── Constants ────────────────────────────────────────────────
MAX_ITERATIONS: int = 3
CONFIDENCE_THRESHOLD: float = 85.0


def run_optimization_loop(
    user_prompt: str,
    complexity: ComplexityLevel,
) -> dict:
    """Run the optimize-evaluate feedback loop until confidence is reached.

    Orchestrates Layer 1 and Layer 2 in a loop:
      - Layer 1 optimizes (or re-optimizes) the prompt
      - Layer 2 scores the optimization and identifies gaps
      - If the score is below threshold, the gaps are fed back to Layer 1
      - Hard cap of MAX_ITERATIONS is always enforced

    Args:
        user_prompt: The user's original prompt (never mutated).
        complexity:  The classified complexity level of the task.

    Returns:
        A dict with keys:
            final_optimized_prompt (str)   - best optimized prompt produced.
            final_confidence_score (float) - score of the final iteration.
            iterations_used        (int)   - how many iterations ran.
            max_iterations_hit     (bool)  - True if loop hit the hard cap.
            task_type              (TaskType) - detected task type.
            all_iterations         (list)  - per-iteration data for benchmarking.
    """
    # ── Detect task type once (doesn't change between iterations) ─
    task_type: TaskType = detect_task_type(user_prompt)

    all_iterations: list[dict] = []
    missing_tasks: list[str] | None = None
    extra_tasks: list[str] | None = None
    final_optimized: str = user_prompt
    final_score: float = 0.0

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n  --- Iteration {iteration}/{MAX_ITERATIONS} ---")

        # ── Layer 1: Optimize ────────────────────────────────
        print(f"  [Layer 1] Optimizing prompt...")
        opt_result = optimize_prompt(
            user_prompt=user_prompt,
            complexity=complexity,
            missing_tasks=missing_tasks,
            extra_tasks=extra_tasks,
            iteration=iteration,
        )

        optimized_prompt = opt_result["optimized_prompt"]

        # ── Layer 2: Evaluate ────────────────────────────────
        print(f"  [Layer 2] Evaluating confidence...")
        eval_result = evaluate_confidence(
            original_prompt=user_prompt,
            optimized_prompt=optimized_prompt,
            task_type=task_type,
            threshold=CONFIDENCE_THRESHOLD,
        )

        confidence_score = eval_result["confidence_score"]
        passes = eval_result["passes_threshold"]

        # ── Record this iteration ────────────────────────────
        iteration_data = {
            "iteration": iteration,
            "optimized_prompt": optimized_prompt,
            "confidence_score": confidence_score,
            "missing_tasks": eval_result["missing_tasks"],
            "extra_tasks": eval_result["extra_tasks"],
        }
        all_iterations.append(iteration_data)

        # ── Update best result ───────────────────────────────
        final_optimized = optimized_prompt
        final_score = confidence_score

        print(f"  Score: {confidence_score}/100 | Threshold: {CONFIDENCE_THRESHOLD}")

        # ── Check exit conditions ────────────────────────────
        if passes:
            print(f"  Threshold reached! Exiting loop.")
            break

        if iteration == MAX_ITERATIONS:
            print(f"  Max iterations reached. Using best available result.")
            break

        # ── Prepare feedback for next iteration ──────────────
        missing_tasks = eval_result["missing_tasks"]
        extra_tasks = eval_result["extra_tasks"]

        if missing_tasks:
            safe = str(missing_tasks).encode("ascii", errors="replace").decode("ascii")
            print(f"  Missing tasks to incorporate: {safe}")
        if extra_tasks:
            safe = str(extra_tasks).encode("ascii", errors="replace").decode("ascii")
            print(f"  Extra tasks to remove: {safe}")

        print(f"  Re-optimizing with feedback...")

    return {
        "final_optimized_prompt": final_optimized,
        "final_confidence_score": final_score,
        "iterations_used": len(all_iterations),
        "max_iterations_hit": len(all_iterations) == MAX_ITERATIONS and not passes,
        "task_type": task_type,
        "all_iterations": all_iterations,
    }


def format_loop_summary(loop_result: dict) -> str:
    """Format the loop result into a human-readable summary string.

    Args:
        loop_result: The dict returned by run_optimization_loop.

    Returns:
        A one-line summary like:
        "Optimization completed in 2 iterations. Final confidence: 92.0/100.
         Threshold reached."
    """
    iterations = loop_result["iterations_used"]
    score = loop_result["final_confidence_score"]
    max_hit = loop_result["max_iterations_hit"]

    status = "Max iterations hit" if max_hit else "Threshold reached"

    return (
        f"Optimization completed in {iterations} iteration(s). "
        f"Final confidence: {score}/100. {status}."
    )


# ── Inline test ──────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    def safe_print(text: str) -> None:
        """Print text safely on Windows terminals that use cp1252."""
        print(text.encode("ascii", errors="replace").decode("ascii"))

    print("=" * 70)
    print("  Iteration Loop - Test")
    print("=" * 70)

    # Moderately complex prompt that should trigger at least 1 re-optimization
    test_prompt = (
        "Build a REST API for a todo app with user authentication, "
        "CRUD operations for tasks, and a PostgreSQL database"
    )

    print(f"\n  Input prompt: \"{test_prompt}\"")
    print(f"  Complexity: HIGH (simulated)")

    result = run_optimization_loop(
        user_prompt=test_prompt,
        complexity=ComplexityLevel.HIGH,
    )

    # ── Print iteration-by-iteration scores ──────────────────
    print(f"\n  {'=' * 50}")
    print(f"  Iteration Summary:")
    print(f"  {'=' * 50}")

    for it in result["all_iterations"]:
        print(f"\n    Iteration {it['iteration']}:")
        print(f"      Score:         {it['confidence_score']}/100")
        print(f"      Missing tasks: {it['missing_tasks']}")
        print(f"      Extra tasks:   {len(it['extra_tasks'])} item(s)")

    print(f"\n  {'=' * 50}")
    print(f"  Final Result:")
    print(f"  {'=' * 50}")
    print(f"    Task type:  {result['task_type'].value}")
    print(f"    Iterations: {result['iterations_used']}")
    print(f"    Max hit:    {result['max_iterations_hit']}")
    print(f"    Score:      {result['final_confidence_score']}/100")
    print(f"\n  Summary: {format_loop_summary(result)}")

    print(f"\n  Final optimized prompt:")
    print(f"  {'-' * 50}")
    safe_print(f"  {result['final_optimized_prompt']}")
    print(f"  {'-' * 50}")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")
