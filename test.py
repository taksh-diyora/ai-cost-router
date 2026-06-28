"""
Quick smoke test for the provider layer.
=========================================
Calls call_llm() once per reasoning role and prints the response.

Usage:
    python test.py

Requires a .env file with valid GROQ_API_KEY and GEMINI_API_KEY.
(ANTHROPIC_API_KEY is optional — Anthropic test is skipped if missing.)
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# ── Load environment variables from .env ─────────────────────
load_dotenv()

# Add project root to path so `src` is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.config.models import ModelRole, get_model
from src.providers import call_llm


def _divider(title: str) -> None:
    """Print a visual divider for readability."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_role(role: ModelRole) -> None:
    """Fire a simple test prompt for a single reasoning role.

    Args:
        role: The ModelRole to test.
    """
    config = get_model(role)
    _divider(f"{role.value.upper()} REASONING  ->  {config.provider}:{config.model_id}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply concisely."},
        {"role": "user", "content": "What are the 5 meanings of life? Reply in 3-4 lines max."},
    ]

    try:
        result = call_llm(role=role, messages=messages, temperature=0.3)

        print(f"  Model:         {result['model']}")
        print(f"  Input tokens:  {result['input_tokens']}")
        print(f"  Output tokens: {result['output_tokens']}")
        print(f"  Response:      {result['content']}")
        print("  [OK]  PASSED")

    except EnvironmentError as exc:
        print(f"  [SKIP]  SKIPPED (no API key): {exc}")

    except RuntimeError as exc:
        print(f"  [FAIL]  FAILED: {exc}")


def main() -> None:
    """Run smoke tests for all three reasoning roles."""
    print("\n[*] AI Cost Router - Provider Smoke Test")
    print("=" * 60)

    for role in ModelRole:
        test_role(role)

    print(f"\n{'=' * 60}")
    print("  Done.  Check results above.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
