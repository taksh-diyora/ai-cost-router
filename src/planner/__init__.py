"""
Planner Router
===============
Selects the appropriate planner (opus or cheap) based on task
complexity and the user's verification choice.

Decision logic:
  - HIGH complexity + ACCEPT_OPTIMIZED  ->  Opus planner (expensive, detailed)
  - Everything else                     ->  Cheap planner (fast, sufficient)
"""

from __future__ import annotations

import os
import sys

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipeline.classifier import ComplexityLevel
from src.pipeline.confidence_evaluator import TaskType
from src.pipeline.user_verification import VerificationChoice
from src.planner.opus_planner import generate_opus_plan
from src.planner.cheap_planner import generate_cheap_plan


def get_plan(
    prompt: str,
    complexity: ComplexityLevel,
    task_type: TaskType,
    user_choice: VerificationChoice,
    request_id: str | None = None,
) -> dict:
    """Route to the appropriate planner based on complexity and user choice.

    Uses the opus planner (HIGH_REASONING) only when the task is
    genuinely complex AND the user accepted the optimized prompt.
    All other cases use the cheap planner to save cost.

    Args:
        prompt:      The prompt to plan for (optimized or original).
        complexity:  The classified complexity level.
        task_type:   The detected task type (CODE, WRITING, etc.).
        user_choice: The user's verification choice.
        request_id:  Optional request ID for logging.

    Returns:
        A dict with the plan result plus a "planner_used" key
        indicating which planner was selected ("opus" or "cheap").
    """
    use_opus = (
        complexity == ComplexityLevel.HIGH
        and user_choice == VerificationChoice.ACCEPT_OPTIMIZED
    )

    if use_opus:
        print("  [Planner] Using OPUS planner (HIGH_REASONING) for complex task.")
        result = generate_opus_plan(prompt, task_type, request_id=request_id)
        result["planner_used"] = "opus"
    else:
        print("  [Planner] Using CHEAP planner (MEDIUM_REASONING).")
        result = generate_cheap_plan(prompt, task_type, request_id=request_id)
        result["planner_used"] = "cheap"

    return result
