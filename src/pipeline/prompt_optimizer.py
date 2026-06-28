"""
Prompt Optimizer (Layer 1)
==========================
Pipeline Stage 4: Rewrites a user's raw prompt into an optimized,
implementation-ready version that produces better results from any
model downstream.

Key principle: PRESERVE the user's intent exactly. Never change
what they want — only make it clearer and more actionable.

On re-optimization (when called from the iteration loop), the
optimizer receives feedback about missing_tasks and extra_tasks
from the Confidence Evaluator (Layer 2) and incorporates them.
"""

from __future__ import annotations

import os
import sys

import tiktoken

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.models import ModelRole
from src.pipeline.classifier import ComplexityLevel
from src.providers import call_llm


# ── Tiktoken encoding (reuse same one as classifier) ────────
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _build_system_prompt(
    complexity: ComplexityLevel,
    missing_tasks: list[str] | None,
    extra_tasks: list[str] | None,
) -> str:
    """Build the system prompt for the optimizer LLM.

    The prompt changes slightly depending on whether this is a
    first-pass optimization or a re-optimization with feedback
    from the Confidence Evaluator.

    Args:
        complexity:    The classified complexity level of the task.
        missing_tasks: Requirements the evaluator found were missing
                       from a previous optimization (None on first pass).
        extra_tasks:   Things the previous optimization added that
                       weren't in the original request (None on first pass).

    Returns:
        The complete system prompt string.
    """
    base = f"""You are an expert prompt engineer. Your job is to take a user's raw prompt and rewrite it into an optimized, clear, implementation-ready version.

The task has been classified as {complexity.value.upper()} complexity.

## Rules — follow these strictly:

1. PRESERVE the original intent exactly. Never add features the user did not ask for. Never remove features they did ask for.
2. Add specificity: if the user did not mention a programming language, framework, or database, make a reasonable assumption and STATE it explicitly (e.g. "Using Python with Flask and SQLite").
3. Break vague requirements into concrete, measurable steps.
4. Remove all ambiguity — every sentence should have one clear interpretation.
5. Structure the output logically: context first, then requirements, then constraints.
6. If the task involves code, specify expected input/output formats, edge cases, and error handling.
7. Do NOT add preamble, explanations, or commentary. Return ONLY the optimized prompt text."""

    # ── Add re-optimization feedback if present ──────────────
    feedback_section = ""

    if missing_tasks:
        missing_list = "\n".join(f"  - {task}" for task in missing_tasks)
        feedback_section += f"""

## CRITICAL — Missing Requirements
The previous optimization was missing these requirements from the original prompt. You MUST incorporate all of them:
{missing_list}"""

    if extra_tasks:
        extra_list = "\n".join(f"  - {task}" for task in extra_tasks)
        feedback_section += f"""

## CRITICAL — Extra Tasks to Remove
The previous optimization incorrectly added these tasks that were NOT in the original prompt. You MUST remove them:
{extra_list}"""

    return base + feedback_section


def optimize_prompt(
    user_prompt: str,
    complexity: ComplexityLevel,
    missing_tasks: list[str] | None = None,
    extra_tasks: list[str] | None = None,
    iteration: int = 1,
) -> dict:
    """Rewrite a user prompt into an optimized, implementation-ready version.

    On the first pass (iteration=1), the optimizer adds specificity and
    structure. On subsequent passes (iteration > 1), it also incorporates
    feedback about missing or extra tasks from the Confidence Evaluator.

    Args:
        user_prompt:   The raw (or previously optimized) prompt to rewrite.
        complexity:    The classified complexity level of the task.
        missing_tasks: Requirements flagged as missing by the evaluator.
        extra_tasks:   Requirements flagged as extra by the evaluator.
        iteration:     Current iteration number (1-based) for loop tracking.

    Returns:
        A dict with keys:
            optimized_prompt (str) — the rewritten prompt text.
            iteration        (int) — which iteration produced this result.

    Raises:
        RuntimeError: If the LLM call fails (propagated from provider).
    """
    system_prompt = _build_system_prompt(complexity, missing_tasks, extra_tasks)

    # ── Build the user message ───────────────────────────────
    if iteration == 1:
        user_message = f"Optimize this prompt:\n\n{user_prompt}"
    else:
        user_message = (
            f"Re-optimize this prompt based on the feedback in your instructions. "
            f"This is iteration {iteration}.\n\n"
            f"Original user prompt:\n{user_prompt}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        result = call_llm(
            role=ModelRole.MEDIUM_REASONING,
            messages=messages,
            temperature=0.4,  # some creativity, but mostly structured
        )

        optimized = result["content"].strip()

        # ── Strip <think>...</think> blocks from reasoning models ─
        if "<think>" in optimized:
            think_end = optimized.rfind("</think>")
            if think_end != -1:
                optimized = optimized[think_end + len("</think>"):].strip()

        return {
            "optimized_prompt": optimized,
            "iteration": iteration,
        }

    except (RuntimeError, EnvironmentError) as exc:
        # If optimization fails, return the original prompt unchanged
        # so the pipeline can still proceed.
        print(f"  [WARN] Prompt optimization failed: {exc}")
        print("  Falling back to original prompt.")
        return {
            "optimized_prompt": user_prompt,
            "iteration": iteration,
        }


def get_token_reduction_estimate(original: str, optimized: str) -> dict:
    """Compare token counts between original and optimized prompts.

    Note: optimized prompts are often LONGER than originals because
    they add specificity, structure, and explicit requirements.
    This is expected and desirable — quality matters more than brevity.

    Args:
        original:  The original user prompt.
        optimized: The optimized prompt from the optimizer.

    Returns:
        A dict with keys:
            original_tokens  (int) — token count of the original.
            optimized_tokens (int) — token count of the optimized.
            difference       (int) — absolute difference in tokens.
            direction        (str) — "reduced" or "expanded".
    """
    original_tokens = len(_ENCODING.encode(original))
    optimized_tokens = len(_ENCODING.encode(optimized))
    difference = abs(original_tokens - optimized_tokens)
    direction = "reduced" if optimized_tokens < original_tokens else "expanded"

    return {
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "difference": difference,
        "direction": direction,
    }


# ── Inline test ──────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    def safe_print(text: str) -> None:
        """Print text safely on Windows terminals that use cp1252."""
        print(text.encode("ascii", errors="replace").decode("ascii"))

    print("=" * 70)
    print("  Prompt Optimizer (Layer 1) - Test")
    print("=" * 70)

    # ── Test 1: First-pass optimization on a vague prompt ────
    vague_prompt = "Make a website for my business"

    print(f"\n  [Test 1] First-pass optimization")
    print(f"  Original: \"{vague_prompt}\"")
    print(f"  Complexity: MEDIUM (simulated)\n")

    result = optimize_prompt(
        user_prompt=vague_prompt,
        complexity=ComplexityLevel.MEDIUM,
    )

    print(f"  Optimized prompt (iteration {result['iteration']}):")
    print(f"  {'-' * 60}")
    safe_print(f"  {result['optimized_prompt']}")
    print(f"  {'-' * 60}")

    # Token comparison
    token_info = get_token_reduction_estimate(vague_prompt, result["optimized_prompt"])
    print(f"\n  Token comparison:")
    print(f"    Original:  {token_info['original_tokens']} tokens")
    print(f"    Optimized: {token_info['optimized_tokens']} tokens")
    print(f"    Direction: {token_info['direction']} by {token_info['difference']} tokens")

    # ── Test 2: Re-optimization with feedback ────────────────
    print(f"\n\n  [Test 2] Re-optimization with evaluator feedback")
    print(f"  Simulating missing_tasks and extra_tasks from Layer 2\n")

    result2 = optimize_prompt(
        user_prompt=vague_prompt,
        complexity=ComplexityLevel.MEDIUM,
        missing_tasks=["Mobile responsive design", "Contact form with email integration"],
        extra_tasks=["Payment processing system"],
        iteration=2,
    )

    print(f"  Re-optimized prompt (iteration {result2['iteration']}):")
    print(f"  {'-' * 60}")
    safe_print(f"  {result2['optimized_prompt']}")
    print(f"  {'-' * 60}")

    # Token comparison: first optimization vs re-optimization
    token_info2 = get_token_reduction_estimate(result["optimized_prompt"], result2["optimized_prompt"])
    print(f"\n  Token comparison (optimized vs re-optimized):")
    print(f"    First optimization:  {token_info2['original_tokens']} tokens")
    print(f"    Re-optimized:        {token_info2['optimized_tokens']} tokens")
    print(f"    Direction: {token_info2['direction']} by {token_info2['difference']} tokens")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")

