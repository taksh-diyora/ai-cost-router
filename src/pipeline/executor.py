"""
Executor
=========
Pipeline Stage 9: Takes a plan (from either planner) and executes it
step by step using a LOW_REASONING model.

The executor processes each step sequentially, passing accumulated
context from previous steps to each new step. This allows a cheap
model to produce coherent output because all the thinking was
already done by the planner.

For LOW complexity tasks, execute_direct() handles single-shot
execution without any plan.
"""

from __future__ import annotations

import os
import re
import sys

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.models import ModelRole
from src.pipeline.confidence_evaluator import TaskType
from src.providers import call_llm


# ── Step parser ──────────────────────────────────────────────

def parse_plan_steps(plan_text: str) -> list[str]:
    """Extract numbered steps from a plan text.

    Handles common formats:
      - "1. Step description"
      - "Step 1: Step description"
      - "1) Step description"
      - "**1.** Step description"

    Multi-line steps are grouped together until the next numbered
    step is detected.

    Args:
        plan_text: The raw plan text from a planner.

    Returns:
        A list of step strings, one per numbered step.
        Returns the full plan as a single step if no numbering is found.
    """
    # ── Pattern: start of line, optional whitespace/bold, digit(s),
    #    followed by . or ) or :, then the step text
    step_pattern = re.compile(
        r"^\s*(?:\*\*)?(\d+)\s*[.):](?:\*\*)?\s*(.*)",
        re.MULTILINE,
    )

    matches = list(step_pattern.finditer(plan_text))

    if not matches:
        # No numbered steps found — return the whole plan as one step
        return [plan_text.strip()] if plan_text.strip() else []

    steps: list[str] = []

    for i, match in enumerate(matches):
        # Start position of this step's content
        step_start = match.start()

        # End position: start of the next step, or end of text
        if i + 1 < len(matches):
            step_end = matches[i + 1].start()
        else:
            step_end = len(plan_text)

        # Extract the full step text (including multi-line content)
        step_text = plan_text[step_start:step_end].strip()
        steps.append(step_text)

    return steps


# ── Step executor ────────────────────────────────────────────

_EXECUTOR_SYSTEM_PROMPT: str = """You are an expert executor. You implement exactly what is asked, nothing more, nothing less. Do not deviate from the plan.

## Rules:
1. Execute ONLY the current step described below
2. Use the provided context from previous steps to maintain continuity
3. Do not skip ahead to future steps
4. Do not revisit completed steps
5. Produce complete, high-quality output for this step
6. If the step involves code, write clean, working code
7. If the step involves writing, produce polished text"""


def execute_step(
    step: str,
    context: str,
    original_prompt: str,
    step_number: int = 1,
) -> dict:
    """Execute a single step from the plan.

    Uses the MEDIUM_REASONING model since the planner already did
    the hard thinking. The executor just follows instructions.

    Args:
        step:            The step text to execute.
        context:         Accumulated output from all previous steps.
        original_prompt: The user's original goal (for reference).
        step_number:     Which step this is (1-based).

    Returns:
        A dict with keys:
            step_output (str) - the output produced by this step.
            step_number (int) - the step number.
    """
    # ── Build user message with full context ─────────────────
    user_message_parts = [
        f"## Original User Goal:\n{original_prompt}",
    ]

    if context.strip():
        user_message_parts.append(
            f"## Output from Previous Steps:\n{context}"
        )

    user_message_parts.append(
        f"## Current Step to Execute (Step {step_number}):\n{step}"
    )

    user_message_parts.append(
        "Execute this step now. Produce the complete output for this step only."
    )

    user_message = "\n\n".join(user_message_parts)

    messages = [
        {"role": "system", "content": _EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        result = call_llm(
            role=ModelRole.MEDIUM_REASONING,
            messages=messages,
            temperature=0.3,
        )

        output = result["content"].strip()

        # ── Strip <think>...</think> blocks ──────────────────
        if "<think>" in output:
            think_end = output.rfind("</think>")
            if think_end != -1:
                output = output[think_end + len("</think>"):].strip()

        return {
            "step_output": output,
            "step_number": step_number,
        }

    except (RuntimeError, EnvironmentError) as exc:
        return {
            "step_output": f"[ERROR] Step {step_number} failed: {exc}",
            "step_number": step_number,
        }


# ── Full plan executor ───────────────────────────────────────

def execute_plan(
    plan: str,
    original_prompt: str,
    task_type: TaskType,
) -> dict:
    """Execute an entire plan step by step.

    Parses the plan into steps and executes them sequentially,
    passing the accumulated output from previous steps as context
    to each subsequent step.

    Args:
        plan:            The full plan text from a planner.
        original_prompt: The user's original prompt (for reference).
        task_type:       The detected task type (for logging).

    Returns:
        A dict with keys:
            final_output          (str)       - the combined output.
            steps_executed        (int)       - how many steps ran.
            step_outputs          (list[dict])- per-step results.
            execution_successful  (bool)      - True if all steps succeeded.
    """
    steps = parse_plan_steps(plan)

    if not steps:
        return {
            "final_output": "[ERROR] No steps found in plan.",
            "steps_executed": 0,
            "step_outputs": [],
            "execution_successful": False,
        }

    print(f"  [Executor] Parsed {len(steps)} steps from plan.")
    print(f"  [Executor] Task type: {task_type.value}")

    step_outputs: list[dict] = []
    accumulated_context: str = ""
    all_succeeded: bool = True

    for i, step in enumerate(steps, start=1):
        print(f"  [Executor] Executing step {i}/{len(steps)}...")

        result = execute_step(
            step=step,
            context=accumulated_context,
            original_prompt=original_prompt,
            step_number=i,
        )

        step_outputs.append(result)

        # ── Check for errors ─────────────────────────────────
        if result["step_output"].startswith("[ERROR]"):
            all_succeeded = False
            print(f"  [Executor] Step {i} FAILED.")
        else:
            print(f"  [Executor] Step {i} completed.")

        # ── Accumulate context for next step ─────────────────
        accumulated_context += f"\n\n--- Step {i} Output ---\n{result['step_output']}"

    # ── Combine all outputs into the final result ────────────
    final_output = "\n\n".join(
        f"--- Step {r['step_number']} ---\n{r['step_output']}"
        for r in step_outputs
    )

    return {
        "final_output": final_output,
        "steps_executed": len(step_outputs),
        "step_outputs": step_outputs,
        "execution_successful": all_succeeded,
    }


# ── Direct executor (for LOW complexity) ─────────────────────

def execute_direct(prompt: str) -> dict:
    """Execute a prompt directly without any plan.

    Used for LOW complexity tasks that skip the planner entirely.
    Single LLM call with LOW_REASONING.

    Args:
        prompt: The prompt to execute directly.

    Returns:
        A dict with keys:
            final_output          (str)  - the LLM's response.
            steps_executed        (int)  - always 1.
            execution_successful  (bool) - True if the call succeeded.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's request clearly and completely."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm(
            role=ModelRole.LOW_REASONING,
            messages=messages,
            temperature=0.4,
        )

        output = result["content"].strip()

        # ── Strip <think>...</think> blocks ──────────────────
        if "<think>" in output:
            think_end = output.rfind("</think>")
            if think_end != -1:
                output = output[think_end + len("</think>"):].strip()

        return {
            "final_output": output,
            "steps_executed": 1,
            "execution_successful": True,
        }

    except (RuntimeError, EnvironmentError) as exc:
        return {
            "final_output": f"[ERROR] Direct execution failed: {exc}",
            "steps_executed": 1,
            "execution_successful": False,
        }


# ── Inline test ──────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    def safe_print(text: str) -> None:
        """Print text safely on Windows terminals that use cp1252."""
        print(text.encode("ascii", errors="replace").decode("ascii"))

    print("=" * 70)
    print("  Executor - Test")
    print("=" * 70)

    # ── Test 1: parse_plan_steps ─────────────────────────────
    print("\n  [Test 1] Parsing plan steps from different formats:")

    sample_plan = """Overview: Build a simple calculator.

1. Define the add function
   - Input: two numbers (a, b)
   - Output: their sum

2. Define the subtract function
   - Input: two numbers (a, b)
   - Output: their difference

3. Create a main function
   - Calls add(5, 3) and subtract(10, 4)
   - Prints both results

Deliverables:
- calculator.py with add, subtract, and main functions"""

    steps = parse_plan_steps(sample_plan)
    print(f"  Steps found: {len(steps)}")
    for i, step in enumerate(steps, 1):
        first_line = step.split("\n")[0][:]
        print(f"    Step {i}: {first_line}...")

    # ── Test 2: Execute the fake plan ────────────────────────
    print(f"\n  [Test 2] Executing the 3-step plan:")

    result = execute_plan(
        plan=sample_plan,
        original_prompt="Build a simple calculator with add and subtract functions",
        task_type=TaskType.CODE,
    )

    print(f"\n  Steps executed: {result['steps_executed']}")
    print(f"  Successful:     {result['execution_successful']}")

    for step_result in result["step_outputs"]:
        print(f"\n  --- Step {step_result['step_number']} Output ---")
        safe_print(step_result["step_output"])

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")
