"""
Shared model access. Every component that calls the model goes through here,
so provider details live in one place (Layer 7 will harden this into a real
provider abstraction with retry/rotation).
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-2.5-flash"


def complete(messages, tools=None, model=None):
    kwargs = {"model": model or MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(**kwargs)
