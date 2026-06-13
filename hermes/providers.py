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


def classify_error(exc) -> str:
    # Map any provider error to a reason, so recovery is a decision not a guess.
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "rate" in name or "429" in msg or "quota" in msg or "resource_exhausted" in msg:
        return "rate_limit"
    if "402" in msg or "billing" in msg or "credit" in msg:
        return "billing"
    if "401" in msg or "403" in msg or "api key" in msg or "unauthorized" in msg:
        return "auth"
    if "context" in msg or "maximum context" in msg or "too long" in msg:
        return "context_overflow"
    if "timeout" in name or "timed out" in msg:
        return "timeout"
    if "500" in msg or "502" in msg or "503" in msg or "overloaded" in msg:
        return "server"
    return "unknown"


# Reason -> what the harness should do about it.
RECOVERY = {
    "rate_limit": "backoff",
    "timeout": "backoff",
    "server": "backoff",
    "context_overflow": "compress",
    "auth": "abort",
    "billing": "abort",
    "unknown": "backoff",
}


def recovery_action(reason: str) -> str:
    return RECOVERY.get(reason, "backoff")
