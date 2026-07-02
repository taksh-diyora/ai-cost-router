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


def detect_user_delegation(user_prompt: str) -> bool:
    """Detect if the user explicitly delegates unspecified details to the model."""
    lower_prompt = user_prompt.lower()
    patterns = [
        "up to you", "up to the next", "up to the model", "up to the llm",
        "you decide", "model decides", "llm decides",
        "don't add", "do not add", "don't ask", "do not ask",
        "everything else", "other things", "other details",
        "leave the rest", "rest is up to"
    ]
    return any(p in lower_prompt for p in patterns)


def _build_system_prompt(
    complexity: ComplexityLevel,
    missing_tasks: list[str] | None,
    extra_tasks: list[str] | None,
    is_delegated: bool = False,
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
    base = f"""You are a prompt optimization specialist. Your job is to rewrite a user's raw
prompt into an optimized, implementation-ready version that is clearer,
more specific, and unambiguous — while preserving the original intent with
absolute fidelity.

Task complexity level: {complexity.value.upper()}

## CLARITY CHECK & SMART BYPASS
Before optimizing, evaluate the quality of the original prompt. 
- Is it already clear, concise, and easy to understand?
- Does it avoid unnecessary repetition?
If the prompt is ALREADY well-written and clear, you must output EXACTLY this string and nothing else: [BYPASS_OPTIMIZATION]
Only proceed with rewriting if the prompt is messy, repetitive, or poorly structured.

## YOUR ONLY JOB:
Clarify WHAT the user wants. Not HOW to build it. Not WHAT TECHNOLOGY to use.
Only: what is the task, what are the requirements, what are the constraints.

## STRICT FORBIDDEN ACTIONS — violating any of these invalidates your output:

FORBIDDEN 1 — Technology injection:
Do NOT suggest, assume, or mention any specific technology, framework, library,
language, or database unless the user explicitly named it in their prompt.
BAD: "Build a REST API using Node.js and Express with a PostgreSQL database"
  (user never said Node.js, Express, or PostgreSQL)
GOOD: "Build a REST API that supports [user's stated requirements]"

FORBIDDEN 2 — Scope expansion:
Do NOT add features, requirements, or steps the user did not ask for.
If the user said "add a login page", do NOT add "with password reset functionality"
unless they asked for it.

FORBIDDEN 3 — Scope reduction:
Do NOT remove, simplify, or skip requirements the user stated. Every requirement
in the original prompt must appear in your output.

FORBIDDEN 4 — Intent changing:
Do NOT reinterpret what the user wants. If they said "sort ascending", do not
say "sort in the desired order". Preserve exact intent.

FORBIDDEN 5 — Adding non-functional requirements:
Do NOT add testing, documentation, deployment, CI/CD, logging, monitoring,
or any operational requirement unless the user explicitly asked for it.

FORBIDDEN 6 — Inferred Best Practices:
NO INFERRED BEST PRACTICES: You must not add standard software development best practices (e.g., error handling, logging, edge-case management, network failure fallbacks) unless the user EXPLICITLY requested them in the original prompt or Q&A context. Stick strictly to the literal features requested.

## REQUIRED ACTIONS:

1. Preserve all original requirements. Every single one.
2. Break vague requirements into concrete, measurable ones.
   VAGUE: "make it fast" → CONCRETE: "operations should complete in under 200ms"
3. Remove ambiguity. Each sentence must have exactly one interpretation.
4. Add specificity only where the user's requirement is genuinely unclear AND
   the clarification does not change the intent.
5. Structure logically: context → requirements → constraints → edge cases.
6. If the user specified a language for a code task: describe expected
   input/output, edge cases, and error handling in concrete terms.
   If no language was specified: describe behavior only, no language.

## OUTPUT FORMAT:
Return ONLY the optimized prompt text. No preamble. No "Here is the optimized
prompt:". No explanation. No commentary. Start directly with the task description."""

    delegation_block = ""
    if is_delegated:
        delegation_block = """

## CRITICAL USER INSTRUCTION — STRICT SCOPE LOCK
The user has explicitly stated that unspecified details should be
left to the next model. This means:
- You may ONLY include what the user literally stated in their prompt.
- You may NOT infer, add, or specify ANY detail the user did not mention.
- Do NOT add: API endpoints, data flows, component descriptions,
  middleware, error handling patterns, edge cases, diagrams, or any
  technical specification the user did not explicitly request.
- Your job is ONLY to clarify what the user DID say, not to fill in
  what they did NOT say.
- If you add ANYTHING beyond the literal user requirements, your
  output is INVALID."""

    base_tail = """

## SELF-CHECK BEFORE RESPONDING:
Ask yourself: "Does my output contain anything the user did NOT ask for?"
If yes → remove it. Ask yourself: "Is anything the user asked for missing?"
If yes → add it back."""

    # ── Add re-optimization feedback if present ──────────────
    feedback_section = ""

    if missing_tasks:
        missing_list = "\n".join(f"{i+1}. {task}" for i, task in enumerate(missing_tasks))
        feedback_section += f"""

## CRITICAL CORRECTION — MISSING REQUIREMENTS
Your previous optimization DROPPED these requirements that were in the original
prompt. This is a FAILURE. You MUST include ALL of the following in your new
output. Do not drop them again:

{missing_list}

Re-read the original prompt carefully and ensure every requirement is present."""

    if extra_tasks:
        extra_list = "\n".join(f"{i+1}. {task}" for i, task in enumerate(extra_tasks))
        feedback_section += f"""

## CRITICAL CORRECTION — HALLUCINATED REQUIREMENTS
Your previous optimization ADDED requirements that the user NEVER asked for.
This is a FAILURE. You MUST remove ALL of the following from your new output:

{extra_list}

You invented these. They are not in the original prompt. Remove them entirely."""

    return base + delegation_block + base_tail + feedback_section


def optimize_prompt(
    user_prompt: str,
    complexity: ComplexityLevel,
    missing_tasks: list[str] | None = None,
    extra_tasks: list[str] | None = None,
    iteration: int = 1,
    request_id: str | None = None,
) -> dict:
    """Rewrite a user prompt into an optimized, implementation-ready version.

    On the first pass (iteration=1), the optimizer adds specificity and
    structure. On subsequent passes (iteration > 1), it also incorporates
    feedback about missing or extra tasks from the Confidence Evaluator.

    Args:
        user_prompt:   The raw (or previously optimized) prompt to rewrite.
        complexity:    The classified complexity level of the task.
        missing_tasks: Requirements flagged as missing by the evaluator.
        extra_tasks:   Tasks the evaluator found were hallucinated.
        iteration:     The current iteration number (1-indexed).
        request_id:    Optional request ID for logging.

    Returns:
        A dict with keys:
            optimized_prompt (str) — the rewritten prompt text.
            iteration        (int) — which iteration produced this result.

    Raises:
        RuntimeError: If the LLM call fails (propagated from provider).
    """
    is_delegated = detect_user_delegation(user_prompt)
    system_prompt = _build_system_prompt(complexity, missing_tasks, extra_tasks, is_delegated)

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
            temperature=0.3,
            request_id=request_id,
            step_name="layer1_optimizer",
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

