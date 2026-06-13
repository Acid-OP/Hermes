"""
Shared model access. The provider is chosen here (default gemini); complete()
wraps every call in retry with jittered exponential backoff so transient
failures (rate limits, timeouts, 5xx) don't crash a run.
"""

import os
import random
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

import providers

load_dotenv()

PROVIDER = providers.get(os.environ.get("HARNESS_PROVIDER", "gemini"))
client = OpenAI(api_key=os.environ[PROVIDER.env_key], base_url=PROVIDER.base_url)
MODEL = PROVIDER.default_model

TRANSIENT = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def _backoff(attempt: int) -> float:
    # Jittered exponential backoff: 1,2,4,8... capped at 60s, plus random jitter
    # so concurrent callers don't retry in lockstep (thundering herd).
    return min(60.0, 2.0**attempt) + random.uniform(0, 1)


def complete(messages, tools=None, model=None, max_retries: int = 5):
    kwargs = {"model": model or MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except TRANSIENT:
            if attempt == max_retries - 1:
                raise
            time.sleep(_backoff(attempt))
