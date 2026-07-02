"""
Confidence Evaluator (Layer 2)
===============================
Pipeline Stage 5: Compares the optimized prompt against the original
to verify that optimization preserved the user's intent.

Produces a weighted confidence score (0-100) based on task-type-specific
criteria. Also identifies missing tasks (things in the original that
were lost) and extra tasks (things added that weren't requested).

This score drives the iteration loop:
  - score >= threshold (default 85) → proceed
  - score < threshold → send feedback to Layer 1 for re-optimization
"""

from __future__ import annotations

import json
import os
import re
import sys
from enum import Enum

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.models import ModelRole
from src.providers import call_llm


# ── Task Type Classification ────────────────────────────────
class TaskType(Enum):
    """Categorises the user's task to select appropriate scoring weights."""

    CODE = "code"
    WRITING = "writing"
    REASONING = "reasoning"
    GENERAL = "general"


# ── Keyword sets for task type detection ─────────────────────
_CODE_KEYWORDS: list[str] = [
    "code", "function", "implement", "bug", "debug", "script",
    "class", "api", "database", "algorithm", "compile", "syntax",
    "import", "library", "module", "endpoint", "backend", "frontend",
    "refactor", "test case", "unit test", "python", "django",
    "architecture", "build", "program",
]

_WRITING_KEYWORDS: list[str] = [
    "write", "essay", "article", "blog", "email", "report",
    "summarize", "draft", "paragraph", "letter", "documentation",
    "content", "copywriting", "poem", "story", "narrative",
]

_REASONING_KEYWORDS: list[str] = [
    "explain", "analyze", "compare", "evaluate", "why",
    "how does", "what is the difference", "reason", "logic",
    "pros and cons", "trade-off", "tradeoff", "cause and effect",
    "argue", "justify", "critique",
]

# ── Default confidence threshold ─────────────────────────────
DEFAULT_THRESHOLD: float = 85.0


# ── Weight configurations per task type ──────────────────────
_WEIGHT_MAP: dict[TaskType, dict[str, int]] = {
    TaskType.CODE: {
        "intent_preserved": 40,
        "completeness": 35,
        "no_hallucinated_requirements": 25,
    },
    TaskType.WRITING: {
        "intent_preserved": 40,
        "key_points_covered": 35,
        "tone_appropriate": 25,
    },
    TaskType.REASONING: {
        "intent_preserved": 50,
        "logical_structure": 30,
        "no_hallucinated_requirements": 20,
    },
    TaskType.GENERAL: {
        "intent_preserved": 50,
        "completeness": 30,
        "clarity_improved": 20,
    },
}


def detect_task_type(user_prompt: str) -> TaskType:
    """Classify the task type using simple keyword matching.

    No LLM call needed — this is a fast, deterministic check
    using keyword presence to pick the best scoring rubric.

    Priority order: CODE > WRITING > REASONING > GENERAL.
    If multiple keyword sets match, the first hit wins.

    Args:
        user_prompt: The user's original prompt text.

    Returns:
        The detected TaskType.
    """
    text_lower = user_prompt.lower()

    # Check each category in priority order
    for keyword in _CODE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
            return TaskType.CODE

    for keyword in _WRITING_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
            return TaskType.WRITING

    for keyword in _REASONING_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
            return TaskType.REASONING

    return TaskType.GENERAL


def get_weights(task_type: TaskType) -> dict[str, int]:
    """Return the scoring weight configuration for a task type.

    Weights always sum to 100 and determine how much each
    criterion contributes to the final confidence score.

    Args:
        task_type: The detected task type.

    Returns:
        A dict mapping criterion names to their integer weights.
    """
    return _WEIGHT_MAP[task_type].copy()


def _parse_evaluation_json(raw_text: str, criteria: list[str]) -> dict:
    """Extract and parse the evaluator LLM's JSON response.

    Handles common LLM output quirks: code fences, thinking tags,
    and partial/malformed JSON. Returns a safe fallback on failure.

    Args:
        raw_text: The raw string returned by the LLM.
        criteria: Expected criterion names for validation.

    Returns:
        Parsed dict with keys: scores, missing_tasks, extra_tasks, reasoning.
    """
    text = raw_text.strip()

    # ── Strip <think>...</think> blocks ──────────────────────
    if "<think>" in text:
        think_end = text.rfind("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()

    # ── Strip markdown code fences ───────────────────────────
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        parsed = json.loads(text)

        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object")

        # ── Validate and extract scores ──────────────────────
        scores = parsed.get("scores", {})
        validated_scores: dict[str, int] = {}
        for criterion in criteria:
            score = scores.get(criterion, 75)  # default 75 if missing
            validated_scores[criterion] = max(0, min(100, int(score)))

        return {
            "scores": validated_scores,
            "missing_tasks": list(parsed.get("missing_tasks", [])),
            "extra_tasks": list(parsed.get("extra_tasks", [])),
            "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
        }

    except (json.JSONDecodeError, ValueError, TypeError):
        # ── Safe fallback: neutral scores, no tasks ──────────
        return {
            "scores": {c: 75 for c in criteria},
            "missing_tasks": [],
            "extra_tasks": [],
            "reasoning": "Evaluation failed — using default scores.",
        }


def evaluate_confidence(
    original_prompt: str,
    optimized_prompt: str,
    task_type: TaskType,
    threshold: float = DEFAULT_THRESHOLD,
    request_id: str | None = None,
) -> dict:
    """Score how well the optimized prompt preserves the original intent.

    Sends both prompts to a MEDIUM_REASONING model with task-type-specific
    scoring criteria. The LLM scores each criterion 0-100, and this
    function computes the final weighted score.

    Args:
        original_prompt:  The user's original prompt.
        optimized_prompt: The rewritten prompt from Layer 1.
        task_type:        Detected task type (determines scoring weights).
        threshold:        Score out of 100 needed to pass.
        request_id:       Optional request ID for logging.

    Returns:
        A dict with keys:
            confidence_score   (float)     - final weighted score 0-100.
            scores             (dict)      - per-criterion scores.
            missing_tasks      (list[str]) - tasks lost during optimization.
            extra_tasks        (list[str]) - tasks added that weren't requested.
            task_type          (TaskType)  - the task type used for scoring.
            reasoning          (str)       - LLM's one-sentence explanation.
            passes_threshold   (bool)      - True if score >= threshold.
    """
    weights = get_weights(task_type)
    criteria = list(weights.keys())

    # ── Build the scoring rubric for the system prompt ───────
    criteria_description = "\n".join(
        f"  - {name} (weight: {weight}%): score 0-100"
        for name, weight in weights.items()
    )

    system_prompt = f"""You are a strict quality auditor. You compare an optimized prompt against the
original and produce a precise confidence score. You are not rewarding effort —
you are measuring accuracy.

Task type detected: {task_type.name}

IMPORTANT: The "Original Prompt" below is the user's TRUE original request. It never changes between iterations. You are always scoring how well the optimized version matches THIS original — not any previous iteration's output.

## SCORING CRITERIA FOR THIS TASK TYPE:
{criteria_description}

## HOW TO SCORE EACH CRITERION (be strict — DO NOT round up):

intent_preserved:
  100 = The optimized prompt requests EXACTLY what the original requested.
  70  = Minor wording change but same meaning.
  40  = Some drift — a requirement was reinterpreted or reframed.
  0   = The core request was changed into something different.

completeness:
  100 = Every single requirement from the original is present in the optimized version.
  70  = One minor requirement is missing.
  40  = A significant requirement is missing.
  0   = Multiple requirements are missing.

no_hallucinated_requirements:
  100 = The optimized prompt contains ONLY what the user asked for.
  70  = One minor extra detail was added (low impact).
  40  = One significant extra requirement was added.
  0   = Multiple requirements were invented.

key_points_covered (writing tasks):
  100 = Every point the user mentioned is addressed.
  50  = At least half the points are addressed.
  0   = Most points are missing.

tone_appropriate (writing tasks):
  100 = The tone and style requested are clearly specified and correct.
  50  = Tone is partially specified or subtly wrong.
  0   = Wrong tone or no tone guidance when user specified one.

logical_structure (reasoning tasks):
  100 = Reasoning flow is clear, ordered, and matches the original structure.
  50  = Some structural drift.
  0   = Structure was completely changed.

clarity_improved (general tasks):
  100 = Objectively clearer than original with no scope change.
  50  = Marginally clearer.
  0   = Same or worse clarity, or clarity was gained by changing scope.

## CALCULATING THE WEIGHTED FINAL SCORE:
Multiply each criterion score by its weight percentage, sum the results.

## IDENTIFYING MISSING AND EXTRA TASKS:
RUTHLESS SCOPE CHECKING: You must aggressively penalize the optimized prompt if it contains ANY features, behaviors, or 'best practices' (like error handling or edge cases) that are not present in the Original Prompt text or the attached Q&A context.
If you detect these unrequested additions, you MUST deduct points and list them explicitly in the `extra_tasks` output list so the optimizer can remove them in the next iteration. A score of 100 is ONLY permitted if the scope is an exact 1:1 match.

missing_tasks: List SPECIFIC requirements from the ORIGINAL that do NOT appear
  in the optimized version. Be precise. Quote the requirement.
  DO NOT list things that are implied or handled by the optimized version.
  Only list things that are genuinely absent.

extra_tasks: List SPECIFIC requirements in the OPTIMIZED version that have NO
  basis in the original. Be precise. Do not list clarifications of existing
  requirements — only list genuinely new requirements that were invented.

## OUTPUT FORMAT — STRICT:
Return ONLY a raw JSON object. No markdown. No backticks. No text before or after.

{{
    "scores": {{{", ".join(f'"{c}": <0-100>' for c in criteria)}}},
    "missing_tasks": ["exact description of missing requirement"],
    "extra_tasks": ["exact description of added requirement that was not requested"],
    "reasoning": "One sentence. State the single most important finding."
}}

If there are no missing tasks, missing_tasks must be [].
If there are no extra tasks, extra_tasks must be [].
DO NOT fabricate missing or extra tasks to fill the arrays."""

    user_message = f"""## Original Prompt:
{original_prompt}

## Optimized Prompt:
{optimized_prompt}

Evaluate the optimization quality. Return ONLY the JSON object."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        result = call_llm(
            role=ModelRole.MEDIUM_REASONING,
            messages=messages,
            temperature=0.0,
            request_id=request_id,
            step_name="layer2_evaluator",
        )

        parsed = _parse_evaluation_json(result["content"], criteria)

    except (RuntimeError, EnvironmentError) as exc:
        print(f"  [WARN] Confidence evaluation failed: {exc}")
        print("  Using default scores.")
        parsed = {
            "scores": {c: 75 for c in criteria},
            "missing_tasks": [],
            "extra_tasks": [],
            "reasoning": f"Evaluation failed: {exc}",
        }

    # ── Calculate weighted confidence score ───────────────────
    confidence_score: float = sum(
        parsed["scores"][criterion] * (weights[criterion] / 100)
        for criterion in criteria
    )

    return {
        "confidence_score": round(confidence_score, 2),
        "scores": parsed["scores"],
        "missing_tasks": parsed["missing_tasks"],
        "extra_tasks": parsed["extra_tasks"],
        "task_type": task_type,
        "reasoning": parsed["reasoning"],
        "passes_threshold": confidence_score >= threshold,
    }


# ── Inline tests ─────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    def safe_print(text: str) -> None:
        """Print text safely on Windows terminals that use cp1252."""
        print(text.encode("ascii", errors="replace").decode("ascii"))

    print("=" * 70)
    print("  Confidence Evaluator (Layer 2) - Test")
    print("=" * 70)

    # ── Test 1: Good optimization (should score high) ────────
    original = "Write a Python function that sorts a list of numbers"
    optimized_good = """Write a Python function called `sort_numbers` that accepts a list of 
integers or floats and returns a new list sorted in ascending order. 
Use Python's built-in sorted() function. Handle edge cases: empty list 
returns empty list, single element returns same list. Include type 
hints and a docstring. Do not modify the original list."""

    task_type = detect_task_type(original)
    weights = get_weights(task_type)

    print(f"\n  [Test 1] Good optimization (should score high)")
    print(f"  Original:  \"{original}\"")
    print(f"  Task type: {task_type.value}")
    print(f"  Weights:   {weights}\n")

    result1 = evaluate_confidence(original, optimized_good, task_type)

    print(f"  Confidence Score: {result1['confidence_score']}")
    print(f"  Passes Threshold: {result1['passes_threshold']}")
    print(f"  Per-criterion scores: {result1['scores']}")
    print(f"  Missing tasks: {result1['missing_tasks']}")
    print(f"  Extra tasks:   {result1['extra_tasks']}")
    safe_print(f"  Reasoning: {result1['reasoning']}")

    # ── Test 2: Bad optimization (should score lower) ────────
    optimized_bad = """Build a full REST API with Flask that includes user authentication, 
database models for products, and a payment processing system 
using Stripe. Deploy to AWS with Docker."""

    print(f"\n  [Test 2] Bad optimization (should score lower)")
    print(f"  Original:  \"{original}\"")
    print(f"  Optimized: completely different task\n")

    result2 = evaluate_confidence(original, optimized_bad, task_type)

    print(f"  Confidence Score: {result2['confidence_score']}")
    print(f"  Passes Threshold: {result2['passes_threshold']}")
    print(f"  Per-criterion scores: {result2['scores']}")
    print(f"  Missing tasks: {result2['missing_tasks']}")
    print(f"  Extra tasks:   {result2['extra_tasks']}")
    safe_print(f"  Reasoning: {result2['reasoning']}")

    # ── Test 3: Task type detection ──────────────────────────
    print(f"\n  [Test 3] Task type detection:")
    test_prompts = [
        ("Implement a binary search algorithm", "CODE"),
        ("Write an essay about climate change", "WRITING"),
        ("Explain how neural networks work", "REASONING"),
        ("Make a website for my business", "GENERAL"),
    ]
    for prompt, expected in test_prompts:
        detected = detect_task_type(prompt)
        status = "[OK]" if detected.value.upper() == expected else "[MISMATCH]"
        print(f"    {status} \"{prompt[:50]}\" -> {detected.value} (expected: {expected})")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")
