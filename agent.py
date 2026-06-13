"""
Agent harness — the loop. Model access is in llm.py, tools in tools.py,
context machinery in context.py.
"""

import hashlib
import json
import sys

import context
import llm
import tools
from tools import ToolCategory

MAX_ITERATIONS = 10


def _idempotency_key(name, args) -> str:
    return hashlib.sha256(
        (name + json.dumps(args, sort_keys=True)).encode()
    ).hexdigest()[:12]


def _approve(tool, args) -> bool:
    if not sys.stdin.isatty():
        print(f"  [APPROVE] auto-approving '{tool.name}' (non-interactive run)")
        return True
    return input(f"  [APPROVE] run {tool.name}({args})? [y/N] ").strip().lower() == "y"


def _execute(t, args) -> str:
    try:
        return str(t.impl(**args))
    except Exception as e:
        return f"ERROR: {e}"


def _maybe_offload(result: str) -> str:
    if len(result) <= 600:
        return result
    log_id = f"log{len(tools.TOOL_LOG) + 1}"
    tools.TOOL_LOG[log_id] = result
    return (
        f"[offloaded {len(result)} chars to {log_id}] "
        f"call read_tool_log('{log_id}') to retrieve the full content."
    )


def run(user_message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a precise assistant. Use tools for math, file reads, and file writes.",
        },
        {"role": "user", "content": user_message},
    ]
    done_actions: dict = {}
    tools.TOOL_LOG.clear()

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- iteration {iteration + 1} (~{context.estimate_tokens(messages)} tok) ---")
        resp = llm.complete(messages, tools=tools.all_schemas())
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
                    result = f"[idempotent] already executed earlier: {done_actions[key]}"
                elif not _approve(t, args):
                    result = "[denied] human declined this action"
                else:
                    result = _execute(t, args)
                    done_actions[key] = result
            else:
                result = _execute(t, args)

            result = _maybe_offload(result)
            print(f"  [OBSERVE] {str(result)[:120]}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            )

    return "Stopped: max iterations reached without a final answer."


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is (128*47)+99? Then read ./note.txt."
    print("USER:", q)
    print("\nANSWER:", run(q))
