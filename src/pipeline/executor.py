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


def split_into_chunks(steps: list, chunk_size: int) -> list[list]:
    """Split a list of steps into smaller chunks."""
    return [steps[i:i + chunk_size] for i in range(0, len(steps), chunk_size)]


# ── Step executor ────────────────────────────────────────────

_EXECUTOR_SYSTEM_PROMPT: str = """You are a step executor. You receive one step from a plan and execute it.
You produce the complete output for that step and nothing else.

## ABSOLUTE RULES — breaking any of these is a failure:

RULE 1 — Execute ONLY the current step.
Do NOT execute future steps. Do NOT reference what future steps will do.
Do NOT produce output for anything other than the current step.

RULE 2 — Use context from previous steps correctly.
The context block contains the accumulated output of all previous steps.
You may reference and build on it. You must maintain continuity.
Do NOT contradict or redo previous steps.

RULE 3 — Produce COMPLETE output for this step.
Do NOT say "I will now...", "Next, we...", or "This step involves...".
Produce the actual output. If the step requires code, write the code.
If the step requires text, write the text. No summaries of what you did.

RULE 4 — Do NOT make decisions not specified in the step.
If the step says to implement a function with a given signature, implement
EXACTLY that function with EXACTLY that signature. Do not rename it.
Do not add parameters. Do not change the return type.

RULE 5 — Do NOT add unrequested elements.
No tests unless specified. No documentation unless specified.
No error handling beyond what the step specifies. No extras."""


_BATCH_EXECUTOR_SYSTEM_PROMPT: str = """You are a batch step executor. You execute an entire implementation plan in one go.
Produce complete, high-quality output for EVERY step without deviating from the plan.
Follow the user's formatting instructions perfectly."""

def parse_batch_output(raw_output: str, num_steps: int, start_step: int = 1) -> list[str]:
    """Parse batched execution output into individual steps.
    
    Extracts content between [STEP_N_START] and [STEP_N_END] tags.
    """
    results = []
    for i in range(start_step, start_step + num_steps):
        start_tag = f"[STEP_{i}_START]"
        end_tag = f"[STEP_{i}_END]"
        
        start_idx = raw_output.find(start_tag)
        end_idx = raw_output.find(end_tag)
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            content = raw_output[start_idx + len(start_tag):end_idx].strip()
            results.append(content)
        else:
            print(f"  [Executor] Warning: Missing or malformed tags for step {i}.")
            results.append("")
    return results

def batch_execute_steps(
    steps: list[str],
    original_prompt: str,
    task_type: TaskType,
    request_id: str | None = None,
    context: str = "",
    start_step: int = 1,
) -> dict:
    """Execute all plan steps in a single LLM call."""
    user_message_parts = [
        f"## Original Goal:\n{original_prompt}\n",
    ]
    if context.strip():
        user_message_parts.append(f"## Output from previous steps:\n{context}\n")
    user_message_parts.extend([
        "## Execute ALL steps below in order. For each step, produce its complete\noutput. You have full context of all steps at once, so you can maintain\nperfect continuity.\n",
        "## Output Format — STRICT:\nFor each step, wrap your output EXACTLY like this:\n[STEP_1_START]\n<complete output for step 1>\n[STEP_1_END]\n[STEP_2_START]\n<complete output for step 2 — may reference step 1's output above>\n[STEP_2_END]\n... and so on for all steps.\n",
        "## Steps to Execute:"
    ])
    for i, step in enumerate(steps, start_step):
        user_message_parts.append(f"Step {i}: {step}")

    user_message = "\n".join(user_message_parts)
    messages = [
        {"role": "system", "content": _BATCH_EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        result = call_llm(
            role=ModelRole.MEDIUM_REASONING,
            messages=messages,
            temperature=0.3,
            request_id=request_id,
            step_name="executor_batch",
        )

        output = result["content"].strip()
        
        # ── Strip <think>...</think> blocks BEFORE parsing ────────
        if "<think>" in output:
            think_end = output.rfind("</think>")
            if think_end != -1:
                output = output[think_end + len("</think>"):].strip()

        parsed_steps = parse_batch_output(output, len(steps), start_step)
        
        empty_indices = [i for i, step in enumerate(parsed_steps) if not step.strip()]
        
        if len(empty_indices) == len(steps):
            raise ValueError("All steps had missing/empty output.")
            
        step_outputs = []
        accumulated_context = context
        all_succeeded = True
        
        if empty_indices:
            print(f"  [Executor] Batch partial recovery: {len(steps) - len(empty_indices)}/{len(steps)} succeeded, re-running {len(empty_indices)} steps sequentially.")
            
        for i, step_content in enumerate(parsed_steps, start_step):
            if not step_content.strip():
                rec_result = execute_step(
                    step=steps[i - start_step],
                    context=accumulated_context,
                    original_prompt=original_prompt,
                    step_number=i,
                    request_id=request_id
                )
                step_content = rec_result["step_output"]
                if step_content.startswith("[ERROR]"):
                    all_succeeded = False
                    
            step_outputs.append({
                "step_output": step_content,
                "step_number": i,
            })
            if len(step_content) // 4 > 500:
                truncated = step_content[:2000] + "...[truncated]"
            else:
                truncated = step_content
            accumulated_context += f"\n\n--- Step {i} Output ---\n{truncated}"
            
        final_output = "\n\n".join(
            f"--- Step {r['step_number']} ---\n{r['step_output']}"
            for r in step_outputs
        )
        
        # ── Token count logging & comparison ──────────────────
        total_output_tokens = result.get("output_tokens", 0)
        avg_step_tokens = total_output_tokens / len(steps) if len(steps) > 0 else 0
        sys_prompt_tokens = 200
        sequential_cost = len(steps) * (sys_prompt_tokens + avg_step_tokens)
        batched_cost = sys_prompt_tokens + (len(steps) * avg_step_tokens)
        savings_pct = (1 - batched_cost / sequential_cost) * 100 if sequential_cost > 0 else 0
        print(f"  [Executor] Mode: batched | Steps: {len(steps)} | Estimated savings vs sequential: ~{savings_pct:.0f}%")

        return {
            "final_output": final_output,
            "steps_executed": len(steps),
            "step_outputs": step_outputs,
            "execution_successful": all_succeeded,
            "execution_mode": "batched",
        }

    except Exception as exc:
        return {
            "final_output": f"[ERROR] Batch execution failed: {exc}",
            "steps_executed": 0,
            "step_outputs": [],
            "execution_successful": False,
            "execution_mode": "batched",
        }


def execute_step(
    step: str,
    context: str,
    original_prompt: str,
    step_number: int = 1,
    request_id: str | None = None,
) -> dict:
    """Execute a single step from the plan.

    Uses the MEDIUM_REASONING model since the planner already did
    the hard thinking. The executor just follows instructions.

    Args:
        step:            The step text to execute.
        context:         Accumulated output from all previous steps.
        original_prompt: The user's original goal (for reference).
        step_number:     Which step this is (1-based).
        request_id:      Optional request ID for logging.

    Returns:
        A dict with keys:
            step_output (str) - the output produced by this step.
            step_number (int) - the step number.
    """
    # ── Build user message with full context ─────────────────
    user_message_parts = [
        f"## Original User Goal (for context only — do NOT re-execute this):\n{original_prompt}",
    ]

    if context.strip():
        user_message_parts.append(
            f"## Output From All Previous Steps (build on this, do not redo it):\n{context}"
        )

    user_message_parts.append(
        f"## Current Step — Step {step_number}:\n{step}"
    )

    user_message_parts.append(
        f"Execute Step {step_number} now. Produce the complete output for this step only.\nDo not summarize. Do not explain what you are doing. Just produce the output."
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
            request_id=request_id,
            step_name=f"executor_step_{step_number}",
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
    request_id: str | None = None,
) -> dict:
    """Execute an entire plan step by step.

    Parses the plan into steps and executes them sequentially,
    passing the accumulated output from previous steps as context
    to each subsequent step.

    Args:
        plan:            The full plan text from a planner.
        original_prompt: The user's original prompt (for reference).
        task_type:       The detected task type (for logging).
        request_id:      Optional request ID for logging.

    Returns:
        A dict with keys:
            final_output          (str)       - the combined output.
            steps_executed        (int)       - how many steps ran.
            step_outputs          (list[dict])- per-step results.
            execution_successful  (bool)      - True if all steps succeeded.
    """
    steps = parse_plan_steps(plan)

    assert steps, "parse_plan_steps returned empty list"
    assert original_prompt.strip(), "original_prompt must not be empty"

    if not steps:
        return {
            "final_output": "[ERROR] No steps found in plan.",
            "steps_executed": 0,
            "step_outputs": [],
            "execution_successful": False,
        }

    print(f"  [Executor] Parsed {len(steps)} steps from plan.")
    print(f"  [Executor] Task type: {task_type.value}")

    if 3 <= len(steps) <= 7:
        batch_result = batch_execute_steps(
            steps=steps,
            original_prompt=original_prompt,
            task_type=task_type,
            request_id=request_id
        )
        if batch_result["execution_successful"]:
            return batch_result
        else:
            print("  [Executor] Warning: Batch execution failed, falling back to sequential")
    elif len(steps) > 7:
        chunks = split_into_chunks(steps, 7)
        accumulated_context = ""
        step_outputs = []
        all_succeeded = True
        
        start_step = 1
        for chunk in chunks:
            batch_result = batch_execute_steps(
                steps=chunk,
                original_prompt=original_prompt,
                task_type=task_type,
                request_id=request_id,
                context=accumulated_context,
                start_step=start_step
            )
            if batch_result["execution_successful"]:
                chunk_outputs = batch_result["step_outputs"]
            else:
                print(f"  [Executor] Warning: Chunk batch execution failed, falling back to sequential for this chunk")
                chunk_outputs = []
                for i, step in enumerate(chunk, start_step):
                    print(f"  [Executor] Executing step {i}/{len(steps)}...")
                    result = execute_step(
                        step=step,
                        context=accumulated_context,
                        original_prompt=original_prompt,
                        step_number=i,
                        request_id=request_id,
                    )
                    chunk_outputs.append(result)
                    if result["step_output"].startswith("[ERROR]"):
                        all_succeeded = False
                    
                    if len(result["step_output"]) // 4 > 500:
                        truncated = result["step_output"][:2000] + "...[truncated]"
                    else:
                        truncated = result["step_output"]
                    accumulated_context += f"\n\n--- Step {i} Output ---\n{truncated}"
            
            if batch_result["execution_successful"]:
                for r in chunk_outputs:
                    if len(r["step_output"]) // 4 > 500:
                        truncated = r["step_output"][:2000] + "...[truncated]"
                    else:
                        truncated = r["step_output"]
                    accumulated_context += f"\n\n--- Step {r['step_number']} Output ---\n{truncated}"
                    
            step_outputs.extend(chunk_outputs)
            start_step += len(chunk)
            
        final_output = "\n\n".join(
            f"--- Step {r['step_number']} ---\n{r['step_output']}"
            for r in step_outputs
        )
        return {
            "final_output": final_output,
            "steps_executed": len(steps),
            "step_outputs": step_outputs,
            "execution_successful": all_succeeded,
            "execution_mode": "batched",
        }

    print(f"  [Executor] Mode: sequential | Steps: {len(steps)}")
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
            request_id=request_id,
        )

        step_outputs.append(result)

        # ── Check for errors ─────────────────────────────────
        if result["step_output"].startswith("[ERROR]"):
            all_succeeded = False
            print(f"  [Executor] Step {i} FAILED.")
        else:
            print(f"  [Executor] Step {i} completed.")

        # ── Accumulate context for next step ─────────────────
        if len(result["step_output"]) // 4 > 500:
            truncated = result["step_output"][:2000] + "...[truncated]"
        else:
            truncated = result["step_output"]
        accumulated_context += f"\n\n--- Step {i} Output ---\n{truncated}"

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
        "execution_mode": "sequential",
    }


# ── Direct executor (for LOW complexity) ─────────────────────

def execute_direct(prompt: str, request_id: str | None = None) -> dict:
    """Execute a prompt directly without any plan.

    Used for LOW complexity tasks that skip the planner entirely.
    Single LLM call with LOW_REASONING.

    Args:
        prompt: The prompt to execute directly.
        request_id: Optional request ID for logging.

    Returns:
        A dict with keys:
            final_output          (str)  - the LLM's response.
            steps_executed        (int)  - always 1.
            execution_successful  (bool) - True if the call succeeded.
    """
    messages = [
        {"role": "system", "content": "You are a direct task executor. Answer or complete the user's request clearly,\ncorrectly, and concisely.\n\nRules:\n1. Answer the question or complete the task directly. No preamble.\n2. Be accurate. Do not guess. If something is uncertain, say so briefly.\n3. Be appropriately concise — not padded, not truncated.\n4. If the task involves code, write working code with no placeholders.\n5. Do not ask follow-up questions. The task is complete enough to execute."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm(
            role=ModelRole.LOW_REASONING,
            messages=messages,
            temperature=0.5,
            request_id=request_id,
            step_name="execute_direct",
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
            "execution_mode": "direct",
        }

    except (RuntimeError, EnvironmentError) as exc:
        return {
            "final_output": f"[ERROR] Direct execution failed: {exc}",
            "steps_executed": 1,
            "execution_successful": False,
            "execution_mode": "direct",
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
