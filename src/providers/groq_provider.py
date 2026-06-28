"""
Groq Provider
=============
Raw SDK wrapper for the Groq API.  Used for LOW and MEDIUM reasoning
tasks during development (free tier).

This module is NEVER imported directly by pipeline code.
All calls go through ``src.providers.call_llm()``.
"""

from __future__ import annotations

import os

from groq import Groq


def call_groq(
    model_id: str,
    messages: list[dict],
    temperature: float = 0.7,
) -> dict:
    """Send a chat-completion request to Groq and return a normalised response.

    Args:
        model_id:    Groq model identifier (e.g. "openai/gpt-oss-20b").
        messages:    OpenAI-style list of {"role": ..., "content": ...} dicts.
        temperature: Sampling temperature (0.0 – 2.0).

    Returns:
        A dict with keys:
            content       (str)  — the assistant's reply text.
            input_tokens  (int)  — prompt token count reported by the API.
            output_tokens (int)  — completion token count reported by the API.
            model         (str)  — the model ID that was actually used.

    Raises:
        EnvironmentError: If GROQ_API_KEY is not set.
        RuntimeError:     If the Groq API call fails for any reason.
    """
    # ── Validate API key ─────────────────────────────────────
    api_key: str | None = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.  "
            "Add it to your .env file or export it in your shell."
        )

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
        )

        # ── Extract and normalise ────────────────────────────
        choice = response.choices[0]
        usage = response.usage

        return {
            "content": choice.message.content,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "model": model_id,
        }

    except Exception as exc:
        raise RuntimeError(
            f"Groq API call failed for model '{model_id}': {exc}"
        ) from exc
