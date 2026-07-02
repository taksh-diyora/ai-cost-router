"""
Opus Planner (High-Reasoning Planner)
======================================
Pipeline Stage 8a: Generates an extremely detailed implementation plan
using the HIGH_REASONING model.

The plan must be SO detailed that a junior developer (or a cheap LLM)
can implement it with ZERO additional thinking. Every step must specify
exact inputs, outputs, function signatures, and error handling.

Used only when:
  - Complexity is HIGH
  - User accepted the optimized prompt
"""

from __future__ import annotations

import os
import sys

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.models import ModelRole, get_model
from src.pipeline.confidence_evaluator import TaskType
from src.providers import call_llm


def _build_opus_system_prompt(task_type: TaskType) -> str:
    """Build the system prompt for the Opus planner.

    The prompt is tailored to the task type so that the planner
    generates the right kind of detail for code, writing, or
    reasoning tasks.

    Args:
        task_type: The detected task type.

    Returns:
        The complete system prompt string.
    """
    # ── Task-type-specific instructions ──────────────────────
    task_instructions = {
        TaskType.CODE: """
## CODE-SPECIFIC REQUIREMENTS:
Every code step must specify:
- The exact function/class/module name
- All parameter names and their types
- The return type and return value structure
- Every error condition and how it is handled (exception type, error message)
- Edge cases and how each is handled
- Any imports required for that step
Do NOT say "write appropriate tests" unless testing was explicitly requested.
Do NOT pick a technology, framework, or library unless the user specified it.""",

        TaskType.WRITING: """
## WRITING-SPECIFIC REQUIREMENTS:
Every writing step must specify:
- The exact section title
- Target word count range (e.g., 150-200 words)
- The 3-5 key points that MUST appear in this section
- The tone (formal/conversational/persuasive/etc.)
- Transition instruction: how this section connects to the next""",

        TaskType.REASONING: """
## REASONING-SPECIFIC REQUIREMENTS:
Every reasoning step must specify:
- The exact logical claim or sub-conclusion to reach in this step
- The specific evidence or premises to use
- The logical operation being performed (induction, deduction, analogy, etc.)
- What the next step will receive as input from this step""",

        TaskType.GENERAL: """
## GENERAL REQUIREMENTS:
- Break the task into clear, actionable steps
- Specify what inputs each step requires and what it produces
- Define quality criteria for each step's output
- Include verification checkpoints""",
    }

    specific_instructions = task_instructions.get(task_type, task_instructions[TaskType.GENERAL])

    return f"""You are an elite implementation planner. Your output will be executed by a
less capable model that has NO ability to make decisions, resolve ambiguity,
or fill in gaps. If your plan has any gap, the executor will fail silently
and produce wrong output. Your plan MUST be complete enough that execution
requires zero intelligence — only mechanical execution of explicit instructions.

## THE ATOMIC STEP TEST — apply to every step before finalizing:
"Could a developer who has NEVER seen the original prompt execute this step
correctly, using ONLY the text in this step plus the outputs of previous steps?"
If the answer is NO for any step → rewrite that step until the answer is YES.

## MANDATORY STRUCTURE — your output must follow this EXACTLY:

### OVERVIEW
One paragraph. State: what the task is, what the final deliverable is,
and what the key constraints are. No vague language.

### STEPS
Numbered steps starting from 1. Each step must contain:
  - ACTION: The exact action to take (verb first, be imperative)
  - INPUT: What this step takes as input (output of step N, or user-provided data)
  - OUTPUT: What this step produces (be specific about format, structure, content)
  - DETAIL: All implementation specifics needed. For code: function signatures,
    data types, error handling approach, edge cases, exact variable names if
    important. For writing: exact section, word count range, key points to cover,
    tone. For reasoning: exact logical steps, what evidence to reference.
  - DONE WHEN: The specific condition that means this step is complete.

## FORBIDDEN IN YOUR PLAN:
- "Handle errors appropriately" → FORBIDDEN. Specify EXACTLY what errors and
  how to handle each one.
- "Use best practices" → FORBIDDEN. State the specific practice.
- "As needed" / "if applicable" / "where appropriate" → ALL FORBIDDEN.
  Every decision must be made by you, not left for the executor.
- "Similar to step N" → FORBIDDEN. Every step must be fully self-contained.
- Any assumption that the executor knows anything about the domain beyond
  what you explicitly state.

{specific_instructions}

### DELIVERABLES CHECKLIST
A bullet list of every artifact the executor must produce. Each item must be
specific enough that the executor can check it off with certainty.
BAD: "• Working code"
GOOD: "• Python function `reverse_string(s: str) -> str` that handles empty
       string, None input, and Unicode characters"

Return ONLY the plan. No preamble, no commentary."""


def generate_opus_plan(prompt: str, task_type: TaskType, request_id: str | None = None) -> dict:
    """Generate an extremely detailed implementation plan.

    Uses the HIGH_REASONING model to produce a plan with enough
    detail that a cheap model can execute it without thinking.

    Args:
        prompt:    The optimized (or original) prompt to plan for.
        task_type: The detected task type (CODE, WRITING, etc.).
        request_id: Optional request ID for logging.

    Returns:
        A dict with keys:
            plan                 (str) - the full plan text.
            step_count           (int) - number of steps detected.
            estimated_complexity (str) - "high" (always for opus).
            model_used           (str) - the model ID used.
    """
    system_prompt = _build_opus_system_prompt(task_type)
    model_config = get_model(ModelRole.HIGH_REASONING)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Create a detailed implementation plan for:\n\n{prompt}"},
    ]

    try:
        result = call_llm(
            role=ModelRole.HIGH_REASONING,
            messages=messages,
            temperature=0.2,
            request_id=request_id,
            step_name="opus_planner",
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
            "estimated_complexity": "high",
            "model_used": model_config.model_id,
        }

    except (RuntimeError, EnvironmentError) as exc:
        return {
            "plan": f"[ERROR] Opus planner failed: {exc}\n\nFallback: Execute the prompt directly.",
            "step_count": 0,
            "estimated_complexity": "high",
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
    print("  Opus Planner (HIGH_REASONING) - Test")
    print("=" * 70)

    test_prompt = (
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
    )

    print(f"\n  Task type: CODE")
    print(f"  Model: {get_model(ModelRole.HIGH_REASONING).model_id}\n")

    result = generate_opus_plan(test_prompt, TaskType.CODE)

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
