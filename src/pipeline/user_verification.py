"""
User Verification
==================
Pipeline Stage 7: After the optimization loop completes, the user
reviews the optimized prompt and decides how to proceed.

Three choices:
  1. Accept the optimized prompt (recommended)
  2. Modify their original prompt and restart the pipeline
  3. Force-use the original prompt as-is (skip optimization)
"""

from __future__ import annotations

import os
import sys
from enum import Enum

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipeline.confidence_evaluator import TaskType


# ── Verification Choice ──────────────────────────────────────
class VerificationChoice(Enum):
    """The user's decision after reviewing the optimized prompt."""

    ACCEPT_OPTIMIZED = "accept_optimized"
    MODIFY_ORIGINAL = "modify_original"
    FORCE_ORIGINAL = "force_original"


def _safe_print(text: str) -> None:
    """Print text safely on Windows terminals that use cp1252.

    LLM-generated content often contains Unicode characters
    (em-dashes, non-breaking hyphens) that crash on Windows.

    Args:
        text: The string to print.
    """
    print(text.encode("ascii", errors="replace").decode("ascii"))


def display_comparison(
    original_prompt: str,
    optimized_prompt: str,
    loop_result: dict,
) -> None:
    """Display the original and optimized prompts side by side.

    Shows a clean, readable comparison so the user can make an
    informed decision about which prompt to proceed with.

    Args:
        original_prompt:  The user's original prompt.
        optimized_prompt: The final optimized prompt from the loop.
        loop_result:      The full dict returned by run_optimization_loop.
    """
    print(f"\n{'=' * 70}")
    print("  === ORIGINAL PROMPT ===")
    print(f"{'=' * 70}")
    _safe_print(original_prompt)

    print(f"\n{'=' * 70}")
    print("  === OPTIMIZED PROMPT ===")
    print(f"{'=' * 70}")
    _safe_print(optimized_prompt)

    print(f"\n{'=' * 70}")
    print("  === OPTIMIZATION SUMMARY ===")
    print(f"{'=' * 70}")

    score = loop_result.get("final_confidence_score", 0)
    iterations = loop_result.get("iterations_used", 0)
    max_hit = loop_result.get("max_iterations_hit", False)
    task_type: TaskType | str = loop_result.get("task_type", "unknown")

    task_type_str = task_type.value if isinstance(task_type, TaskType) else str(task_type)

    print(f"  Confidence Score:    {score}/100")
    print(f"  Iterations Used:     {iterations}")
    print(f"  Max Iterations Hit:  {max_hit}")
    print(f"  Task Type Detected:  {task_type_str}")
    print(f"{'=' * 70}\n")


def get_user_choice() -> VerificationChoice:
    """Prompt the user to choose how to proceed.

    Displays three numbered options and validates input.
    Re-prompts on invalid input until a valid choice is made.

    Returns:
        The selected VerificationChoice enum value.
    """
    choice_map = {
        "1": VerificationChoice.ACCEPT_OPTIMIZED,
        "2": VerificationChoice.MODIFY_ORIGINAL,
        "3": VerificationChoice.FORCE_ORIGINAL,
    }

    print("  How would you like to proceed?\n")
    print("    1. Continue with optimized prompt (recommended)")
    print("    2. Modify my original prompt and restart")
    print("    3. Use my original prompt as-is")
    print()

    while True:
        raw = input("  Enter your choice (1/2/3): ").strip()

        if raw in choice_map:
            return choice_map[raw]

        print(f"  Invalid input: '{raw}'. Please enter 1, 2, or 3.")


def _collect_multiline_input() -> str:
    """Collect multi-line input from the user.

    The user types their modified prompt and enters END on a
    new line by itself to finish.

    Returns:
        The collected text as a single string (without the END marker).
    """
    print("  Please enter your modified prompt.")
    print("  (Type END on a new line by itself when finished)\n")

    lines: list[str] = []

    while True:
        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    return "\n".join(lines)


def handle_verification(
    original_prompt: str,
    loop_result: dict,
) -> dict:
    """Run the full verification flow: display, choose, act.

    Orchestrates the display, user choice, and any follow-up
    actions (like collecting a modified prompt).

    Args:
        original_prompt: The user's original prompt.
        loop_result:     The full dict returned by run_optimization_loop.

    Returns:
        A dict with keys:
            prompt_to_use (str)               - the prompt to proceed with.
            choice        (VerificationChoice) - what the user chose.
            restart       (bool)              - True if pipeline should restart
                                                from the classifier with a new prompt.
    """
    optimized_prompt = loop_result.get("final_optimized_prompt", original_prompt)

    # ── Show comparison ──────────────────────────────────────
    display_comparison(original_prompt, optimized_prompt, loop_result)

    # ── Get user decision ────────────────────────────────────
    choice = get_user_choice()

    # ── Act on choice ────────────────────────────────────────
    if choice == VerificationChoice.ACCEPT_OPTIMIZED:
        print("\n  [OK] Proceeding with the optimized prompt.\n")
        return {
            "prompt_to_use": optimized_prompt,
            "choice": choice,
            "restart": False,
        }

    elif choice == VerificationChoice.MODIFY_ORIGINAL:
        print()
        modified_prompt = _collect_multiline_input()

        if not modified_prompt.strip():
            print("  [WARN] Empty input received. Using original prompt instead.")
            return {
                "prompt_to_use": original_prompt,
                "choice": choice,
                "restart": True,
            }

        print(f"\n  [OK] Restarting pipeline with your modified prompt.\n")
        return {
            "prompt_to_use": modified_prompt,
            "choice": choice,
            "restart": True,
        }

    else:  # FORCE_ORIGINAL
        print("\n  [OK] Using the original prompt as-is. Skipping optimization.\n")
        return {
            "prompt_to_use": original_prompt,
            "choice": choice,
            "restart": False,
        }


# ── Inline test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  User Verification - Test")
    print("=" * 70)

    # Simulate a fake loop_result
    fake_loop_result = {
        "final_optimized_prompt": (
            "Create a RESTful API for a todo application that includes "
            "user authentication and task management, using PostgreSQL "
            "for data storage.\n\n"
            "Requirements:\n"
            "1. User registration and login with token-based auth\n"
            "2. CRUD operations for tasks (create, read, update, delete)\n"
            "3. Each task linked to the authenticated user\n"
            "4. JSON request/response format with proper HTTP status codes\n"
            "5. Input validation with clear error messages\n\n"
            "Constraints:\n"
            "- PostgreSQL for all persistence\n"
            "- All task endpoints require authentication\n"
            "- RESTful conventions, stateless API"
        ),
        "final_confidence_score": 97.5,
        "iterations_used": 1,
        "max_iterations_hit": False,
        "task_type": TaskType.CODE,
        "all_iterations": [
            {
                "iteration": 1,
                "optimized_prompt": "...",
                "confidence_score": 97.5,
                "missing_tasks": [],
                "extra_tasks": [],
            }
        ],
    }

    original = (
        "Build a REST API for a todo app with user authentication, "
        "CRUD operations for tasks, and a PostgreSQL database"
    )

    print("\n  (This is an interactive test. Choose an option when prompted.)\n")

    result = handle_verification(original, fake_loop_result)

    print(f"  --- Result ---")
    print(f"  Choice:  {result['choice'].value}")
    print(f"  Restart: {result['restart']}")
    print(f"  Prompt:  {result['prompt_to_use'][:100]}...")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")
