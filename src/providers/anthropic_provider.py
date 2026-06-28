"""
Anthropic Provider
==================
Raw SDK wrapper for the Anthropic API.  Used for final production
testing with Claude Sonnet and Claude Opus.

This module is NEVER imported directly by pipeline code.
All calls go through ``src.providers.call_llm()``.
"""

from __future__ import annotations

import os

from anthropic import Anthropic


def call_anthropic(
    model_id: str,
    messages: list[dict],
    temperature: float = 0.7,
) -> dict:
    """Send a chat-completion request to Anthropic and return a normalised response.

    The Anthropic SDK expects a ``system`` parameter separate from the
    ``messages`` list.  This function extracts any "system" role message
    automatically so callers can use the standard OpenAI-style format.

    Args:
        model_id:    Anthropic model identifier (e.g. "claude-sonnet-4-6").
        messages:    OpenAI-style list of {"role": ..., "content": ...} dicts.
        temperature: Sampling temperature (0.0 – 1.0 for Anthropic).

    Returns:
        A dict with keys:
            content       (str)  — the assistant's reply text.
            input_tokens  (int)  — prompt token count reported by the API.
            output_tokens (int)  — completion token count reported by the API.
            model         (str)  — the model ID that was used.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
        RuntimeError:     If the Anthropic API call fails for any reason.
    """
    # ── Validate API key ─────────────────────────────────────
    api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.  "
            "Add it to your .env file or export it in your shell."
        )

    try:
        client = Anthropic(api_key=api_key)

        # ── Separate system message from conversation ────────
        # Anthropic's API takes `system` as a top-level kwarg,
        # not as part of the messages list.
        system_text: str | None = None
        chat_messages: list[dict] = []

        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                chat_messages.append(msg)

        # ── Build API kwargs ─────────────────────────────────
        api_kwargs: dict = {
            "model": model_id,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system_text:
            api_kwargs["system"] = system_text

        # ── Call the API ─────────────────────────────────────
        response = client.messages.create(**api_kwargs)

        # ── Extract and normalise ────────────────────────────
        # Anthropic returns a list of content blocks; concatenate text blocks.
        content_text: str = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return {
            "content": content_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model_id,
        }

    except Exception as exc:
        raise RuntimeError(
            f"Anthropic API call failed for model '{model_id}': {exc}"
        ) from exc
