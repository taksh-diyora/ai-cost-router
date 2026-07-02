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
        TaskType.CODE: "(CODE)      This is a coding task. Include: specific function/class names,\n            input/output types, error handling approach, and key logic per step.",
        TaskType.WRITING: "(WRITING)   This is a writing task. Include: section names, target length,\n            key points per section, and tone.",
        TaskType.REASONING: "(REASONING) This is a reasoning/analysis task. Include: each logical step,\n            what conclusion it reaches, and what it passes to the next step.",
        TaskType.GENERAL: "(GENERAL)   Structure steps as concrete actions with clear inputs and outputs.",
    }

    hint = task_hint.get(task_type, task_hint[TaskType.GENERAL])

    return f"""You are a task planner. Create a structured, step-by-step implementation plan
for the given task. Your plan will be executed by a model that follows
instructions literally — be explicit, not suggestive.

## RULES:
1. Number every step. Start from 1.
2. Each step is ONE concrete action. Not a category. Not a phase. One action.
   BAD: "Step 3: Handle authentication"
   GOOD: "Step 3: Create a function validate_token(token: str) that checks
          the token against the database and returns the user ID if valid,
          or raises an InvalidTokenError if not."
3. Each step must state what it takes as input and what it produces as output.
4. Do NOT use vague language: "appropriate", "as needed", "handle properly",
   "best practices". State the specific action.
5. Do NOT add requirements the user did not ask for.
6. End with a deliverables checklist — a bulleted list of every artifact
   the executor must produce to consider the task complete.

{hint}

## OUTPUT FORMAT:
Return ONLY the plan. No preamble. No "Here is my plan:". Start with Step 1."""


def generate_cheap_plan(prompt: str, task_type: TaskType, request_id: str | None = None) -> dict:
    """Generate a structured step-by-step plan.

    Uses the MEDIUM_REASONING model to produce a plan with
    clear steps. Less detailed than the opus plan but sufficient
    for most tasks.

    Args:
        prompt:    The optimized (or original) prompt to plan for.
        task_type: The detected task type (CODE, WRITING, etc.).
        request_id: Optional request ID for logging.

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
            request_id=request_id,
            step_name="cheap_planner",
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
