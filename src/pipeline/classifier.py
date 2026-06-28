"""
Hybrid Complexity Classifier
=============================
Pipeline Stage 3: Determines if a task is LOW, MEDIUM, or HIGH complexity.

Uses a hybrid approach:
  1. Rule-based classification first (zero LLM cost)
  2. LLM fallback ONLY when rules cannot confidently decide

This saves money by avoiding LLM calls for obvious cases like
simple questions (LOW) or massive implementation requests (HIGH).
"""

from __future__ import annotations

import os
import re
import sys
from enum import Enum

import tiktoken

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.models import ModelRole
from src.providers import call_llm


# ── Complexity Levels ────────────────────────────────────────
class ComplexityLevel(Enum):
    """Task complexity tier that determines model routing downstream."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Keyword lists for rule-based classification ──────────────
# Presence of these keywords suggests the task is NOT trivially simple.
_COMPLEXITY_KEYWORDS: list[str] = [
    "implement", "integrate", "architect", "design", "build",
    "create", "develop", "system", "database", "api",
    "refactor", "migrate", "optimize", "deploy", "scale",
    "authentication", "authorization",
]

# Subset of keywords that signal HIGH complexity specifically.
_HIGH_COMPLEXITY_KEYWORDS: list[str] = [
    "implement", "architect", "system", "integrate", "database",
    "deploy", "scale", "migrate", "authentication", "microservice",
    "pipeline", "infrastructure",
]

# ── Token counter ────────────────────────────────────────────
# Load the encoding once at module level (cl100k_base is used by
# GPT-4 / GPT-3.5 and gives a reasonable approximation for any model).
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate the token count of a string without calling any API.

    Uses tiktoken's cl100k_base encoding, which is the tokenizer for
    GPT-4 and gives a close-enough estimate for cost/complexity purposes.

    Args:
        text: The input string to tokenize.

    Returns:
        Number of tokens in the string.
    """
    return len(_ENCODING.encode(text))


def _count_keyword_matches(text: str, keyword_list: list[str]) -> int:
    """Count how many keywords from a list appear in the text.

    Uses word-boundary regex matching so 'api' doesn't match inside
    'capital' and 'scale' doesn't match inside 'escalate'.

    Args:
        text:         The text to search (will be lowercased internally).
        keyword_list: List of keywords to look for.

    Returns:
        Number of distinct keywords found.
    """
    text_lower = text.lower()
    return sum(
        1 for kw in keyword_list
        if re.search(rf"\b{re.escape(kw)}\b", text_lower)
    )


def _has_code_blocks(text: str) -> bool:
    """Check if the text contains triple-backtick code blocks.

    Args:
        text: The text to check.

    Returns:
        True if at least one ``` code fence is found.
    """
    return "```" in text


def rule_based_classify(user_prompt: str) -> ComplexityLevel | None:
    """Attempt to classify complexity using deterministic rules.

    Checks token count, keyword presence, and code blocks to decide
    complexity WITHOUT any LLM call. Returns None if the prompt
    falls into an ambiguous zone that rules can't confidently handle.

    Classification Logic:
        LOW:  <200 tokens AND no complexity keywords AND no code blocks
        HIGH: >1500 tokens OR 3+ high-complexity keywords OR
              (code blocks AND implementation keywords)
        MEDIUM: 200-1500 tokens with 1-2 complexity keywords
        None:  anything that doesn't clearly fit

    Args:
        user_prompt: The user's raw prompt text.

    Returns:
        A ComplexityLevel if rules can decide, or None for ambiguous cases.
    """
    tokens = count_tokens(user_prompt)
    complexity_kw_count = _count_keyword_matches(user_prompt, _COMPLEXITY_KEYWORDS)
    high_kw_count = _count_keyword_matches(user_prompt, _HIGH_COMPLEXITY_KEYWORDS)
    has_code = _has_code_blocks(user_prompt)

    # ── LOW: simple, short, no technical keywords, no code ───
    if tokens < 200 and complexity_kw_count == 0 and not has_code:
        return ComplexityLevel.LOW

    # ── HIGH: any of these conditions is enough ──────────────
    if tokens > 1500:
        return ComplexityLevel.HIGH
    if high_kw_count >= 3:
        return ComplexityLevel.HIGH
    if has_code and complexity_kw_count >= 1:
        return ComplexityLevel.HIGH

    # ── MEDIUM: moderate length with some keywords ───────────
    if 200 <= tokens <= 1500 and 1 <= complexity_kw_count <= 2:
        return ComplexityLevel.MEDIUM

    # ── AMBIGUOUS: rules can't decide confidently ────────────
    return None


# ── LLM system prompt for classification ─────────────────────
_CLASSIFY_SYSTEM_PROMPT: str = """You are a task complexity classifier. Given a task description, classify it as exactly one of: LOW, MEDIUM, or HIGH.

Definitions:
- LOW: Simple questions, lookups, short translations, basic explanations, trivial code snippets. Can be handled by any model.
- MEDIUM: Moderate tasks like writing functions, summarizing documents, fixing bugs with context, multi-step reasoning. Needs a decent model.
- HIGH: Complex tasks like system design, multi-file implementation, architecture decisions, large refactors, building full features. Needs the best model.

Rules:
1. Respond with ONLY one word: LOW, MEDIUM, or HIGH
2. No explanation, no punctuation, no extra text
3. Just the single word"""


def llm_classify(user_prompt: str) -> ComplexityLevel:
    """Classify complexity by asking a MEDIUM_REASONING model.

    This is the fallback path — only called when rule_based_classify
    returns None (ambiguous case). Uses a low-cost model since
    classification itself is a simple task.

    Args:
        user_prompt: The user's raw prompt text.

    Returns:
        A ComplexityLevel. Defaults to MEDIUM if the LLM response
        cannot be parsed into a valid level.
    """
    messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify this task:\n\n{user_prompt}"},
    ]

    try:
        result = call_llm(
            role=ModelRole.MEDIUM_REASONING,
            messages=messages,
            temperature=0.1,  # near-deterministic for classification
        )

        # ── Parse the response ───────────────────────────────
        raw = result["content"].strip().upper()

        # Strip <think>...</think> blocks from reasoning models
        if "<think>" in raw.lower():
            think_end = raw.lower().rfind("</think>")
            if think_end != -1:
                raw = raw[think_end + len("</think>"):].strip()

        # Try to match a valid level anywhere in the response
        for level in ComplexityLevel:
            if level.value.upper() in raw:
                return level

        # If no match found, default to MEDIUM (safest middle ground)
        print(f"  [WARN] Could not parse LLM classification: '{raw}'. Defaulting to MEDIUM.")
        return ComplexityLevel.MEDIUM

    except (RuntimeError, EnvironmentError) as exc:
        print(f"  [WARN] LLM classification failed: {exc}. Defaulting to MEDIUM.")
        return ComplexityLevel.MEDIUM


def classify(user_prompt: str) -> dict:
    """Classify a user prompt's complexity using the hybrid approach.

    Main entry point for the classifier. Tries rule-based first
    (free, instant), falls back to LLM only when rules can't decide.

    Args:
        user_prompt: The user's raw prompt text.

    Returns:
        A dict with keys:
            complexity  (ComplexityLevel) - LOW, MEDIUM, or HIGH
            method      (str)            - "rule_based" or "llm"
            token_count (int)            - token count of the prompt
    """
    token_count = count_tokens(user_prompt)

    # ── Try rule-based first (zero cost) ─────────────────────
    rule_result = rule_based_classify(user_prompt)

    if rule_result is not None:
        return {
            "complexity": rule_result,
            "method": "rule_based",
            "token_count": token_count,
        }

    # ── Fall back to LLM (only for ambiguous cases) ──────────
    llm_result = llm_classify(user_prompt)

    return {
        "complexity": llm_result,
        "method": "llm",
        "token_count": token_count,
    }


# ── Inline tests ─────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    test_prompts = [
        # ── Should be LOW (rule-based) ───────────────────────
        (
            "What is the capital of France?",
            "Simple factual question — should be LOW via rules",
        ),
        (
            "Explain what a variable is in Python",
            "Basic knowledge question — should be LOW via rules",
        ),

        # ── Should be HIGH (rule-based) ──────────────────────
        (
            "Architect a microservice system with authentication, "
            "integrate a PostgreSQL database, and deploy it to Kubernetes "
            "with auto-scaling infrastructure",
            "Multiple high-complexity keywords — should be HIGH via rules",
        ),
        (
            "```python\ndef login():\n    pass\n```\n\nImplement this login function "
            "with proper authentication and session management",
            "Code block + implementation keywords — should be HIGH via rules",
        ),

        # ── Should be MEDIUM (rule-based) ─────────────────────
        (
            "I need you to help me with a project. " + " ".join(["Here is some additional context about the project that provides more background information and details."] * 15)
            + " Please implement a sorting algorithm for this use case.",
            "260 tokens with 1 complexity keyword — should be MEDIUM via rules",
        ),

        # ── Ambiguous (should trigger LLM) ───────────────────
        (
            "I want you to take my resume and rewrite it completely from scratch. "
            + " ".join(["Make it sound more professional, fix all grammar issues, "
            "restructure the sections completely, and tailor it specifically "
            "for a senior software engineering role at a top tech company."] * 7),
            "245 tokens, zero complexity keywords — should fall to LLM",
        ),

        # ── Should be LOW (rule-based) — verifying short prompts ─
        (
            "Write me a good cover letter for a software engineering "
            "position at Google. I have 3 years of experience with "
            "Python, React, and AWS.",
            "Short, no complexity keywords — correctly LOW via rules",
        ),
    ]

    print("=" * 70)
    print("  Hybrid Complexity Classifier - Test")
    print("=" * 70)

    for i, (prompt, description) in enumerate(test_prompts, start=1):
        result = classify(prompt)
        display_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt

        print(f"\n  Test {i}: {description}")
        print(f"  Prompt:     \"{display_prompt}\"")
        print(f"  Complexity: {result['complexity'].value.upper()}")
        print(f"  Method:     {result['method']}")
        print(f"  Tokens:     {result['token_count']}")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")
