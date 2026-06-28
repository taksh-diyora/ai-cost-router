"""
Missing Info Detector
=====================
Pipeline Stage 2: Checks if a user's prompt has missing or ambiguous
information that would prevent proper execution.

Uses a LOW_REASONING model to analyze the prompt and identify gaps,
then loops with follow-up questions until the prompt is complete
(or a hard cap of 3 rounds is hit).
"""

from __future__ import annotations

import json
import os
import sys

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.models import ModelRole
from src.providers import call_llm


# ── Constants ────────────────────────────────────────────────
_MAX_ROUNDS: int = 3

_SYSTEM_PROMPT: str = """You are an expert requirements analyst. Your job is to determine if a user's prompt contains enough information to be executed properly.

Analyze the prompt and identify any missing, ambiguous, or unclear information that would prevent a developer or AI from executing the task correctly.

## Examples of what counts as missing info:

- "Implement a login system" -> MISSING: Which programming language? Which framework? Which database for storing credentials? Should it include OAuth or just email/password?
- "Fix my code" -> MISSING: What is the code? What is wrong with it? What is the expected behavior? What language is it in?
- "Build me a REST API" -> MISSING: What resources/endpoints? Which language/framework? What authentication method? What database?

## Examples of prompts that ARE complete enough:

- "Write an essay about AI" -> This is complete enough. The user wants a general essay, no specifics are strictly required.
- "Explain how TCP/IP works" -> This is complete enough. It is a knowledge question with a clear scope.
- "Write a Python function that reverses a string" -> This is complete enough. Language, task, and scope are all clear.

## Rules:
1. Do NOT flag prompts as incomplete just because they could have MORE detail. Only flag genuinely ambiguous or missing critical information.
2. Be practical — if a reasonable default exists, the info is not "missing".
3. Return your analysis as a JSON object with EXACTLY this structure (no markdown, no backticks, just raw JSON):

{
    "is_complete": true/false,
    "missing_info": ["list of specific things that are unclear or missing"],
    "follow_up_questions": ["exact questions to ask the user to fill in the gaps"]
}

If the prompt is complete, return:
{
    "is_complete": true,
    "missing_info": [],
    "follow_up_questions": []
}"""


# ── Safe fallback when LLM returns bad JSON ─────────────────
_SAFE_FALLBACK: dict = {
    "is_complete": True,
    "missing_info": [],
    "follow_up_questions": [],
}


def _parse_llm_json(raw_text: str) -> dict:
    """Extract and parse JSON from an LLM response string.

    Handles common LLM quirks:
      - Response wrapped in ```json ... ``` code fences
      - Leading/trailing whitespace or newlines
      - Thinking tags from reasoning models (e.g. <think>...</think>)

    Args:
        raw_text: The raw string returned by the LLM.

    Returns:
        Parsed dict if valid JSON is found, otherwise the safe fallback.
    """
    text = raw_text.strip()

    # ── Strip <think>...</think> blocks from reasoning models ─
    if "<think>" in text:
        think_end = text.rfind("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()

    # ── Strip markdown code fences ───────────────────────────
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        parsed = json.loads(text)

        # ── Validate expected keys exist ─────────────────────
        if not isinstance(parsed, dict):
            return _SAFE_FALLBACK.copy()
        if "is_complete" not in parsed:
            return _SAFE_FALLBACK.copy()

        # ── Ensure correct types with defaults ───────────────
        return {
            "is_complete": bool(parsed.get("is_complete", True)),
            "missing_info": list(parsed.get("missing_info", [])),
            "follow_up_questions": list(parsed.get("follow_up_questions", [])),
        }

    except (json.JSONDecodeError, ValueError):
        return _SAFE_FALLBACK.copy()


def detect_missing_info(user_prompt: str) -> dict:
    """Analyze a user prompt for missing or ambiguous information.

    Sends the prompt to a LOW_REASONING model with a specialized system
    prompt that asks it to identify gaps in the requirements.

    Args:
        user_prompt: The raw prompt submitted by the user.

    Returns:
        A dict with keys:
            is_complete        (bool)       - True if the prompt is ready.
            missing_info       (list[str])  - What is unclear or missing.
            follow_up_questions (list[str]) - Questions to ask the user.

        On any error (LLM failure, bad JSON), returns a safe fallback
        that marks the prompt as complete so the pipeline continues.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this prompt for missing information:\n\n{user_prompt}"},
    ]

    try:
        result = call_llm(
            role=ModelRole.MEDIUM_REASONING,
            messages=messages,
            temperature=0.3,  # low temp for structured output
        )
        return _parse_llm_json(result["content"])

    except (RuntimeError, EnvironmentError) as exc:
        # If the LLM call fails, don't block the pipeline —
        # treat the prompt as complete and let it proceed.
        print(f"  [WARN] Missing info detection failed: {exc}")
        return _SAFE_FALLBACK.copy()


def run_missing_info_loop(user_prompt: str) -> str:
    """Interactively resolve missing information in a user prompt.

    Calls detect_missing_info in a loop. If the prompt is incomplete,
    prints follow-up questions, collects user answers, appends them
    to the prompt, and re-checks. Stops when:
      - The prompt is marked as complete, OR
      - 3 rounds of questions have been asked (hard cap).

    Args:
        user_prompt: The raw prompt submitted by the user.

    Returns:
        The original prompt (if already complete) or an enriched version
        with the user's additional context appended.
    """
    current_prompt: str = user_prompt

    for round_num in range(1, _MAX_ROUNDS + 1):
        print(f"\n--- Missing Info Check (round {round_num}/{_MAX_ROUNDS}) ---")

        analysis = detect_missing_info(current_prompt)

        if analysis["is_complete"]:
            print("  Prompt is complete. No missing information detected.")
            return current_prompt

        # ── Show what's missing ──────────────────────────────
        print("\n  Missing information detected:")
        for item in analysis["missing_info"]:
            print(f"    - {item}")

        # ── Ask follow-up questions ──────────────────────────
        if not analysis["follow_up_questions"]:
            # LLM flagged missing info but gave no questions — skip
            print("  No follow-up questions generated. Proceeding with current prompt.")
            return current_prompt

        print("\n  Please answer the following questions:\n")
        answers: list[str] = []

        for i, question in enumerate(analysis["follow_up_questions"], start=1):
            print(f"  Q{i}: {question}")
            answer = input(f"  A{i}: ").strip()
            if answer:
                answers.append(f"Q: {question}\nA: {answer}")

        # ── Append answers as additional context ─────────────
        if answers:
            additional_context = "\n\n".join(answers)
            current_prompt = (
                f"{current_prompt}\n\n"
                f"--- Additional Context (Round {round_num}) ---\n"
                f"{additional_context}"
            )
            print("\n  Prompt enriched with your answers. Re-checking...")
        else:
            print("\n  No answers provided. Proceeding with current prompt.")
            return current_prompt

    # ── Hard cap reached ─────────────────────────────────────
    print(f"\n  [INFO] Maximum {_MAX_ROUNDS} rounds reached. Proceeding with current prompt.")
    return current_prompt


# ── Inline test ──────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Intentionally vague prompt to trigger follow-up questions
    vague_prompt = "Build me a login system"

    print("=" * 60)
    print("  Missing Info Detector - Test")
    print("=" * 60)

    # Test 1: detect_missing_info (single call, no loop)
    print("\n[Test 1] Single detection on vague prompt:")
    print(f"  Prompt: \"{vague_prompt}\"")
    result = detect_missing_info(vague_prompt)
    print(f"  is_complete: {result['is_complete']}")
    print(f"  missing_info: {result['missing_info']}")
    print(f"  follow_up_questions: {result['follow_up_questions']}")

    # Test 2: detect_missing_info on a clear prompt
    clear_prompt = "Write a Python function that checks if a number is prime"
    print(f"\n[Test 2] Single detection on clear prompt:")
    print(f"  Prompt: \"{clear_prompt}\"")
    result = detect_missing_info(clear_prompt)
    print(f"  is_complete: {result['is_complete']}")
    print(f"  missing_info: {result['missing_info']}")
    print(f"  follow_up_questions: {result['follow_up_questions']}")

    # Test 3: Full interactive loop (will ask for user input)
    print(f"\n[Test 3] Interactive loop on vague prompt:")
    print(f"  Starting prompt: \"{vague_prompt}\"")
    print("  (Answer the follow-up questions when prompted)\n")
    final_prompt = run_missing_info_loop(vague_prompt)
    print(f"\n  Final enriched prompt:\n{final_prompt}")
