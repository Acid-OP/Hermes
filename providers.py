"""
Layer 7 — Reliability: provider abstraction + the resilience layer.

The model is swappable: each provider is described by its OpenAI-compatible
endpoint, env key, default model, and context window. Switch with one env var;
the loop never changes. This module also owns error classification (below) so
retry/recovery decisions live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Provider:
    name: str
    base_url: str
    env_key: str
    default_model: str
    context_window: int = 1_000_000


PROVIDERS = {
    "gemini": Provider(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
        1_000_000,
    ),
    "openai": Provider(
        "openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o", 128_000
    ),
    "openrouter": Provider(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "openai/gpt-4o",
        128_000,
    ),
}


def get(name: str) -> Provider:
    return PROVIDERS[name]
