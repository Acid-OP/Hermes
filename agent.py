"""
Agent harness — the loop. Tools and their registry live in tools.py.
"""

import hashlib
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

import tools
from tools import ToolCategory

load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-2.5-flash"
MAX_ITERATIONS = 10


def _idempotency_key(name, args) -> str:
    return hashlib.sha256(
        (name + json.dumps(args, sort_keys=True)).encode()
    ).hexdigest()[:12]


def _approve(tool, args) -> bool:
    # Human-in-the-loop gate for irreversible ACTION tools. Pausing here is a
    # different kind of stop than Layer 1's: the loop pauses because it reached
    # the edge of its authority, not because the task is done.
    if not sys.stdin.isatty():
        print(f"  [APPROVE] auto-approving '{tool.name}' (non-interactive run)")
        return True
    return input(f"  [APPROVE] run {tool.name}({args})? [y/N] ").strip().lower() == "y"


def _execute(t, args) -> str:
    try:
        return str(t.impl(**args))
    except Exception as e:
        # Tool failure is an observation, never an exception that kills the loop.
        return f"ERROR: {e}"


def run(user_message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a precise assistant. Use tools for math, file reads, and file writes.",
        },
        {"role": "user", "content": user_message},
    ]
    done_actions: dict = {}  # idempotency ledger for side-effecting calls

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- iteration {iteration + 1} ---")
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools.all_schemas()
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [ACT]     {name}({args})")

            t = tools.get(name)
            if t is None:
                result = f"ERROR: unknown tool '{name}'"
            elif t.category == ToolCategory.ACTION:
                key = _idempotency_key(name, args)
                if key in done_actions:
                    # A retried side effect is recognized, not repeated. This is
                    # what stops an action (a payment, a message) from firing twice.
                    result = f"[idempotent] already executed earlier: {done_actions[key]}"
                elif not _approve(t, args):
                    result = "[denied] human declined this action"
                else:
                    result = _execute(t, args)
                    done_actions[key] = result
            else:
                result = _execute(t, args)

            print(f"  [OBSERVE] {str(result)[:120]}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            )

    return "Stopped: max iterations reached without a final answer."


if __name__ == "__main__":
    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is (128*47)+99? Then read ./note.txt and tell me what it says."
    )
    print("USER:", q)
    print("\nANSWER:", run(q))
