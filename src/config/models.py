"""
Model Configuration — Single Source of Truth
=============================================
This is the ONLY place in the entire project where model names,
provider choices, and per-token costs are defined.

Every other module imports from here via `get_model(role)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ── Model Roles ──────────────────────────────────────────────
class ModelRole(Enum):
    """Classifies the reasoning capability required for a task."""

    LOW_REASONING = "low"
    MEDIUM_REASONING = "medium"
    HIGH_REASONING = "high"


# ── Model Configuration ─────────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for a single LLM endpoint.

    Attributes:
        provider:           Provider name — one of "groq", "gemini", "anthropic".
        model_id:           The exact model identifier passed to the provider SDK.
        cost_per_1k_input:  Cost in USD per 1 000 input tokens.
        cost_per_1k_output: Cost in USD per 1 000 output tokens.
    """

    provider: str
    model_id: str
    cost_per_1k_input: float
    cost_per_1k_output: float


# ── Active Model Map (dev configuration) ────────────────────
# During development we use free-tier providers:
#   LOW / MEDIUM  → Groq   (free API)
#   HIGH          → Gemini (free tier, 50 req/day on pro)
#
# Costs are set to 0.0 for free-tier models.
# Swap to Anthropic models + real costs for production testing.

MODELS: dict[ModelRole, ModelConfig] = {
    ModelRole.LOW_REASONING: ModelConfig(
        provider="gemini",
        model_id="gemini-2.5-flash",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
    ModelRole.MEDIUM_REASONING: ModelConfig(
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
    ModelRole.HIGH_REASONING: ModelConfig(
        provider="groq",
        model_id="qwen/qwen3-32b",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
}


def get_model(role: ModelRole) -> ModelConfig:
    """Look up the active model configuration for a given reasoning role.

    Args:
        role: The reasoning tier required (LOW, MEDIUM, or HIGH).

    Returns:
        The ModelConfig registered for that role.

    Raises:
        ValueError: If the role has no registered model.
    """
    try:
        return MODELS[role]
    except KeyError:
        raise ValueError(
            f"No model configured for role '{role.value}'. "
            f"Available roles: {[r.value for r in ModelRole]}"
        )
