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

_SYSTEM_PROMPT: str = """You are a requirements completeness checker. Your only job is to determine
whether a prompt has enough information to be executed by a competent developer
or AI model WITHOUT asking any clarifying questions.

## THE ONLY QUESTION YOU MUST ANSWER:
"Could a competent developer start working on this task RIGHT NOW and produce
something correct?" If YES → is_complete: true. If NO → is_complete: false.

## WHEN TO FLAG AS INCOMPLETE (missing_info must be critical, not optional):
- No language/runtime specified AND the choice fundamentally changes the solution
  (e.g., "implement a login system" with no language → flag)
- No context given when context is literally required to do the task
  (e.g., "fix my code" with no code attached → flag)
- The task contradicts itself or is logically impossible as stated
- Two equally valid interpretations exist that produce COMPLETELY different outputs

## WHEN NOT TO FLAG (do NOT invent missing info):
- A reasonable default exists for the missing detail → NOT missing
  (e.g., "write a REST API" → assume JSON, HTTP, standard conventions → complete)
- The detail is a style preference, not a functional requirement → NOT missing
  (e.g., "use tabs or spaces?" → not missing)
- The user is asking a knowledge question → always complete
  (e.g., "Explain how TCP/IP works" → complete)
- The user wants general creative work → almost always complete
  (e.g., "Write an essay about AI" → complete)
- More specificity would be NICE but is not REQUIRED to start → NOT missing

## ABSOLUTE RULES:
1. DO NOT flag something as missing just because you personally want more detail.
2. DO NOT flag something as missing if a competent developer would make a
   reasonable assumption and proceed.
3. If you are uncertain whether to flag → DO NOT flag. Err toward is_complete: true.
4. ACCEPT NEGATIVE CONSTRAINTS: If a user answers a clarification question with 'no', 'none', 'not needed', or 'no specification', you MUST treat that missing information as RESOLVED and SATISFIED. Do not ask the user for that specific information again in subsequent rounds.
5. follow_up_questions must be SPECIFIC and ANSWERABLE, not vague.
   BAD: "What technology stack do you want to use?"
   GOOD: "What programming language should the login system be written in?"

## OUTPUT FORMAT — STRICT:
Return ONLY a raw JSON object. No markdown. No backticks. No explanation.
No text before or after the JSON. The JSON must match this schema exactly:

{
    "is_complete": true or false,
    "missing_info": ["specific thing that is unclear"],
    "follow_up_questions": ["exact question to ask the user"]
}

If is_complete is true, missing_info and follow_up_questions MUST be empty arrays [].
If is_complete is false, both arrays MUST have at least one item each.

---
### EXAMPLES OF HANDLING NEGATIVE ANSWERS
If a user answers a question indicating they don't care, have no preference, or want to skip it, you MUST treat that requirement as completely satisfied. Do NOT ask about it again.

Example 1:
Q: What database system should be used?
User Answer: none
Your Action: Treat the database requirement as SATISFIED. Do not ask about a database again.

Example 2:
Q: Should the system use OAuth or OpenID Connect?
User Answer: no specification
Your Action: Treat the authentication protocol requirement as SATISFIED. Do not ask about authentication protocols again.

Example 3:
Q: What password hashing algorithm should be used?
User Answer: doesn't matter
Your Action: Treat the hashing algorithm requirement as SATISFIED. Do not ask about algorithms again.
---"""


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


def detect_missing_info(user_prompt: str, request_id: str | None = None) -> dict:
    """Analyze a user prompt for missing or ambiguous information.

    Sends the prompt to a LOW_REASONING model with a specialized system
    prompt that asks it to identify gaps in the requirements.

    Args:
        user_prompt: The raw prompt submitted by the user.
        request_id:  Optional request ID for benchmark logging.

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
            temperature=0.1,
            request_id=request_id,
            step_name="missing_info_detector",
        )
        return _parse_llm_json(result["content"])

    except (RuntimeError, EnvironmentError) as exc:
        # If the LLM call fails, don't block the pipeline —
        # treat the prompt as complete and let it proceed.
        print(f"  [WARN] Missing info detection failed: {exc}")
        return _SAFE_FALLBACK.copy()


def run_missing_info_loop(user_prompt: str, request_id: str | None = None) -> str:
    """Interactively resolve missing information in a user prompt.

    Calls detect_missing_info in a loop. If the prompt is incomplete,
    prints follow-up questions, collects user answers, appends them
    to the prompt, and re-checks. Stops when:
      - The prompt is marked as complete, OR
      - 3 rounds of questions have been asked (hard cap).

    Args:
        user_prompt: The raw prompt submitted by the user.
        request_id:  Optional request ID for logging.

    Returns:
        The original prompt (if already complete) or an enriched version
        with the user's additional context appended.
    """
    current_prompt: str = user_prompt

    for round_num in range(1, _MAX_ROUNDS + 1):
        print(f"\n--- Missing Info Check (round {round_num}/{_MAX_ROUNDS}) ---")

        analysis = detect_missing_info(current_prompt, request_id=request_id)

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
        skip_phrases = {"no specification", "none", "no", "doesn't matter", "not needed", "skip", "na", "n/a"}

        for i, question in enumerate(analysis["follow_up_questions"], start=1):
            print(f"  Q{i}: {question}")
            answer = input(f"  A{i}: ").strip()
            if answer and answer.lower() not in skip_phrases:
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
