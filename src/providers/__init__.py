"""
Unified LLM Provider Interface
===============================
Exposes a single ``call_llm()`` function that the entire pipeline uses.

**No other module should import groq_provider / gemini_provider /
anthropic_provider directly.**  This gateway reads the model config,
picks the right provider, and guarantees a consistent return shape.
"""

from __future__ import annotations

from src.config.models import ModelRole, ModelConfig, get_model
from src.providers.groq_provider import call_groq
from src.providers.gemini_provider import call_gemini
from src.providers.anthropic_provider import call_anthropic


# ── Provider dispatch table ─────────────────────────────────
_PROVIDER_FN = {
    "groq": call_groq,
    "gemini": call_gemini,
    "anthropic": call_anthropic,
}


def call_llm(
    role: ModelRole,
    messages: list[dict],
    temperature: float = 0.7,
) -> dict:
    """Route an LLM request to the correct provider based on reasoning role.

    This is the ONLY function the rest of the codebase calls to talk to
    any LLM.  It resolves the provider + model_id from ``config.models``
    and delegates to the matching SDK wrapper.

    Args:
        role:        The reasoning tier required (LOW, MEDIUM, or HIGH).
        messages:    OpenAI-style list of {"role": ..., "content": ...} dicts.
        temperature: Sampling temperature passed through to the provider.

    Returns:
        A dict with keys:
            content       (str)  — the model's reply text.
            input_tokens  (int)  — prompt tokens reported by the API.
            output_tokens (int)  — completion tokens reported by the API.
            model         (str)  — the model ID that was used.

    Raises:
        ValueError:  If the role or provider is not configured.
        RuntimeError: Propagated from the underlying provider on API errors.
    """
    # ── Resolve model config ─────────────────────────────────
    config: ModelConfig = get_model(role)

    # ── Look up provider function ────────────────────────────
    provider_fn = _PROVIDER_FN.get(config.provider)
    if provider_fn is None:
        raise ValueError(
            f"Unknown provider '{config.provider}' for role '{role.value}'.  "
            f"Supported providers: {list(_PROVIDER_FN.keys())}"
        )

    # ── Delegate to the provider ─────────────────────────────
    return provider_fn(
        model_id=config.model_id,
        messages=messages,
        temperature=temperature,
    )
