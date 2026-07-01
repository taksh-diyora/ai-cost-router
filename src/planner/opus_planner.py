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
## Code-Specific Requirements:
- Specify exact file names and directory structure
- Write complete function signatures with parameter types and return types
- Define all data structures (classes, dicts, schemas) with field names and types
- For each function: state its purpose, inputs, outputs, and error cases
- Specify the error handling approach (try/except, custom exceptions, error codes)
- List all edge cases that must be handled
- Define expected input/output formats with concrete examples""",

        TaskType.WRITING: """
## Writing-Specific Requirements:
- Provide a detailed outline with section headers
- Specify approximate word count for each section
- Define the tone (formal, casual, technical, persuasive) for each section
- List the key points that MUST appear in each paragraph
- Specify the target audience and reading level
- Include transition guidance between sections""",

        TaskType.REASONING: """
## Reasoning-Specific Requirements:
- Break down each logical step explicitly
- Specify what evidence or data to reference at each step
- Define the reasoning method (deductive, inductive, comparative, causal)
- State what conclusions should be drawn at each checkpoint
- Identify potential counterarguments and how to address them
- Specify how to structure the final conclusion""",

        TaskType.GENERAL: """
## General Task Requirements:
- Break the task into clear, actionable steps
- Specify what inputs each step requires and what it produces
- Define quality criteria for each step's output
- Include verification checkpoints""",
    }

    specific_instructions = task_instructions.get(task_type, task_instructions[TaskType.GENERAL])

    return f"""You are an elite planning agent. Your job is to generate an implementation plan SO detailed and precise that a junior developer (or a cheap, less capable AI model) can execute it with ZERO additional thinking or decision-making.

## Core Principles:
1. Every step must be NUMBERED and self-contained
2. Each step must specify EXACTLY what to do, what inputs it takes, and what outputs it produces
3. Leave NOTHING to interpretation -- if any step is unclear, your plan has FAILED
4. Do not skip "obvious" steps -- spell out everything explicitly
5. Use concrete examples, not abstract descriptions
{specific_instructions}

## Output Format:
Structure your plan as:
1. **Overview**: One paragraph summarizing the entire task
2. **Steps**: Numbered steps (1, 2, 3...) with full detail
3. **Deliverables Checklist**: A bullet-point list of every artifact that must be produced

Return ONLY the plan. No preamble, no commentary."""


def generate_opus_plan(prompt: str, task_type: TaskType) -> dict:
    """Generate an extremely detailed implementation plan.

    Uses the HIGH_REASONING model to produce a plan with enough
    detail that a cheap model can execute it without thinking.

    Args:
        prompt:    The optimized (or original) prompt to plan for.
        task_type: The detected task type (CODE, WRITING, etc.).

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
            temperature=0.3,  # low temp for precise, structured output
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
