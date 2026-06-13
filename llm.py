"""
Shared model access. The provider is chosen here (default gemini) so the rest
of the harness never hardcodes one. Layer 7 adds retry/backoff around complete().
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

import providers

load_dotenv()

PROVIDER = providers.get(os.environ.get("HARNESS_PROVIDER", "gemini"))
client = OpenAI(api_key=os.environ[PROVIDER.env_key], base_url=PROVIDER.base_url)
MODEL = PROVIDER.default_model


def complete(messages, tools=None, model=None):
    kwargs = {"model": model or MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(**kwargs)
