"""
Layer 1 — The Agent Loop (the inner engine).

This is the whole engine of an agent: assemble context -> reason -> act ->
observe -> repeat -> stop. Everything in later layers (tools, context,
memory, harness) hangs off THIS loop. Read the run() function bottom-up.

Wired to Gemini via its OpenAI-compatible endpoint, so the code is the
canonical `messages` + `tool_calls` shape used by every harness in the wild.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# TOOLS  (Layer 2 will go deep here; for now, two toy tools to drive the loop)
# A tool is two things: a Python function, and a JSON schema the MODEL sees.
# ---------------------------------------------------------------------------


def calculator(expression: str) -> str:
    """Toy calculator. eval() is unsafe — fine for a lab, never in prod."""
    return str(eval(expression))


def read_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()[:2000]
    except Exception as e:  # tool errors become DATA, never exceptions (see loop)
        return f"ERROR: {e}"


TOOLS_IMPL = {"calculator": calculator, "read_file": read_file}

# The schema is how the model decides WHAT to call and with WHICH args.
# Descriptions are written FOR THE MODEL, not for you — they steer behavior.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a Python arithmetic expression like '2*(3+4)'. Use for ALL math; never compute in your head.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from disk, returns up to 2000 chars.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# THE LOOP
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 10  # STOP CONDITION #1: the hard guard against runaway loops


def run(user_message: str) -> str:
    # `messages` IS the agent's entire state. The loop body is stateless;
    # all memory of the run lives in this growing list. This is the single
    # most important idea in Layer 1.
    messages = [
        {
            "role": "system",
            "content": "You are a precise assistant. Use tools for math and file reads. Never guess a calculation.",
        },
        {"role": "user", "content": user_message},
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- iteration {iteration + 1} ---")

        # REASON: one model call. The model sees all of `messages` + the tools,
        # and decides: call tool(s), or produce a final answer.
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA
        )
        msg = resp.choices[0].message

        # Append the assistant's turn to state (it may carry tool_calls).
        # Echoing it back verbatim next iteration is what gives continuity.
        messages.append(msg.model_dump(exclude_none=True))

        # STOP CONDITION #2 — the natural terminator: no tool calls means the
        # model is done acting and produced a final message.
        # NOTE: "no tool calls" != "goal achieved". The model could be wrong or
        # asking a question. A real harness verifies the goal (Layer 5). For
        # now, terminal message = done.
        if not msg.tool_calls:
            return msg.content

        # ACT + OBSERVE: run each requested tool, feed the result back as a
        # `tool` message keyed by tool_call_id, then loop so the model can
        # reason over the result.
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [ACT]     {name}({args})")
            try:
                result = TOOLS_IMPL[name](**args)
            except Exception as e:
                # Tool failure is fed back as an observation, NOT raised. This
                # lets the model see the error and self-correct next iteration.
                result = f"ERROR: {e}"
            print(f"  [OBSERVE] {str(result)[:120]}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            )

    # STOP CONDITION #3: the guard fired. We never let the model spin forever.
    return "Stopped: max iterations reached without a final answer."


if __name__ == "__main__":
    import sys

    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is (128*47) + 99? Then read ./note.txt and tell me what it says."
    )
    print("USER:", q)
    print("\nANSWER:", run(q))
