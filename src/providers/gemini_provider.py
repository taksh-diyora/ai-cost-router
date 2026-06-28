"""
Google Gemini Provider
======================
Raw SDK wrapper for the Google Generative AI (Gemini) API.
Used for HIGH reasoning tasks during development (free tier).

This module is NEVER imported directly by pipeline code.
All calls go through ``src.providers.call_llm()``.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types


def _convert_messages_to_gemini(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """Convert OpenAI-style messages into Gemini's native format.

    Gemini uses a different conversation structure:
      - A separate ``system`` instruction (string).
      - A list of ``Content`` objects with role "user" or "model".

    Args:
        messages: OpenAI-format list of {"role": ..., "content": ...}.

    Returns:
        A tuple of (system_instruction, contents).
    """
    system_instruction: str | None = None
    contents: list[types.Content] = []

    for msg in messages:
        role = msg["role"]
        text = msg["content"]

        if role == "system":
            # Gemini accepts a single system instruction string
            system_instruction = text
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
        elif role == "assistant":
            # Gemini calls the assistant role "model"
            contents.append(types.Content(role="model", parts=[types.Part(text=text)]))

    return system_instruction, contents


def call_gemini(
    model_id: str,
    messages: list[dict],
    temperature: float = 0.7,
) -> dict:
    """Send a chat request to Google Gemini and return a normalised response.

    Internally converts the OpenAI-style ``messages`` list into Gemini's
    Content format so that the rest of the codebase can use a single
    message schema everywhere.

    Args:
        model_id:    Gemini model identifier (e.g. "gemini-2.5-flash").
        messages:    OpenAI-style list of {"role": ..., "content": ...} dicts.
        temperature: Sampling temperature (0.0 – 2.0).

    Returns:
        A dict with keys:
            content       (str)  — the model's reply text.
            input_tokens  (int)  — prompt token count reported by the API.
            output_tokens (int)  — completion token count reported by the API.
            model         (str)  — the model ID that was used.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
        RuntimeError:     If the Gemini API call fails for any reason.
    """
    # ── Validate API key ─────────────────────────────────────
    api_key: str | None = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.  "
            "Add it to your .env file or export it in your shell."
        )

    try:
        # ── Build the client ─────────────────────────────────
        client = genai.Client(api_key=api_key)

        # ── Convert message format ───────────────────────────
        system_instruction, contents = _convert_messages_to_gemini(messages)

        # ── Build generation config ──────────────────────────
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
        )
        if system_instruction:
            gen_config.system_instruction = system_instruction

        # ── Call the API ─────────────────────────────────────
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=gen_config,
        )

        # ── Extract and normalise ────────────────────────────
        content_text: str = response.text or ""

        # Token counts from usage_metadata
        usage = response.usage_metadata
        input_tokens: int = usage.prompt_token_count if usage and usage.prompt_token_count else 0
        output_tokens: int = usage.candidates_token_count if usage and usage.candidates_token_count else 0

        return {
            "content": content_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model_id,
        }

    except Exception as exc:
        raise RuntimeError(
            f"Gemini API call failed for model '{model_id}': {exc}"
        ) from exc
