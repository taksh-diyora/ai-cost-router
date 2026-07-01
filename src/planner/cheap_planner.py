"""
Cheap Planner (Medium-Reasoning Planner)
=========================================
Pipeline Stage 8b: Generates a structured step-by-step plan using
the MEDIUM_REASONING model.

Simpler than the Opus planner -- provides clear steps without the
extreme granularity. Sufficient for LOW and MEDIUM complexity tasks,
and used as the default planner for most requests.

Used when:
  - Complexity is LOW or MEDIUM
  - User forced original prompt
  - Any case that doesn't qualify for the opus planner
"""

from __future__ import annotations

import os
import sys

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.models import ModelRole, get_model
from src.pipeline.confidence_evaluator import TaskType
from src.providers import call_llm


def _build_cheap_system_prompt(task_type: TaskType) -> str:
    """Build the system prompt for the cheap planner.

    Simpler than the opus prompt -- focuses on clear structure
    and actionable steps without extreme granularity.

    Args:
        task_type: The detected task type.

    Returns:
        The complete system prompt string.
    """
    task_hint = {
        TaskType.CODE: "This is a coding task. Include file structure, key functions, and technologies to use.",
        TaskType.WRITING: "This is a writing task. Include an outline, key points, and tone guidance.",
        TaskType.REASONING: "This is a reasoning task. Include logical steps and evidence to reference.",
        TaskType.GENERAL: "Break this into clear, actionable steps.",
    }

    hint = task_hint.get(task_type, task_hint[TaskType.GENERAL])

    return f"""You are a planning agent. Create a clear, structured, step-by-step plan for the given task.

{hint}

## Rules:
1. Number every step
2. Each step should be a concrete action, not a vague instruction
3. Include what inputs each step needs and what it produces
4. End with a short checklist of deliverables

Return ONLY the plan. No preamble, no commentary."""


def generate_cheap_plan(prompt: str, task_type: TaskType) -> dict:
    """Generate a structured step-by-step plan.

    Uses the MEDIUM_REASONING model to produce a plan with
    clear steps. Less detailed than the opus plan but sufficient
    for most tasks.

    Args:
        prompt:    The optimized (or original) prompt to plan for.
        task_type: The detected task type (CODE, WRITING, etc.).

    Returns:
        A dict with keys:
            plan                 (str) - the full plan text.
            step_count           (int) - number of steps detected.
            estimated_complexity (str) - "low" or "medium".
            model_used           (str) - the model ID used.
    """
    system_prompt = _build_cheap_system_prompt(task_type)
    model_config = get_model(ModelRole.MEDIUM_REASONING)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Create a plan for:\n\n{prompt}"},
    ]

    try:
        result = call_llm(
            role=ModelRole.MEDIUM_REASONING,
            messages=messages,
            temperature=0.3,
        )

        plan_text = result["content"].strip()

        # ── Strip <think>...</think> blocks ──────────────────
        if "<think>" in plan_text:
            think_end = plan_text.rfind("</think>")
            if think_end != -1:
                plan_text = plan_text[think_end + len("</think>"):].strip()

        # ── Count steps (lines starting with a number followed by .)
        step_count = sum(
            1 for line in plan_text.split("\n")
            if line.strip() and line.strip()[0].isdigit() and "." in line.strip()[:5]
        )

        return {
            "plan": plan_text,
            "step_count": max(step_count, 1),
            "estimated_complexity": "medium",
            "model_used": model_config.model_id,
        }

    except (RuntimeError, EnvironmentError) as exc:
        return {
            "plan": f"[ERROR] Cheap planner failed: {exc}\n\nFallback: Execute the prompt directly.",
            "step_count": 0,
            "estimated_complexity": "medium",
            "model_used": model_config.model_id,
        }


# ── Inline test ──────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    def safe_print(text: str) -> None:
        """Print text safely on Windows terminals that use cp1252."""
        print(text.encode("ascii", errors="replace").decode("ascii"))

    print("=" * 70)
    print("  Cheap Planner (MEDIUM_REASONING) - Test")
    print("=" * 70)

    test_prompt = (
        "Write a Python script that reads a CSV file, filters rows "
        "where the 'status' column equals 'active', and writes the "
        "filtered results to a new CSV file."
    )

    print(f"\n  Task type: CODE")
    print(f"  Model: {get_model(ModelRole.MEDIUM_REASONING).model_id}\n")

    result = generate_cheap_plan(test_prompt, TaskType.CODE)

    print(f"  Steps detected: {result['step_count']}")
    print(f"  Model used:     {result['model_used']}")
    print(f"  Complexity:     {result['estimated_complexity']}")
    print(f"\n  {'=' * 50}")
    print(f"  Plan:")
    print(f"  {'=' * 50}")
    safe_print(result["plan"])
    print(f"  {'=' * 50}")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")
