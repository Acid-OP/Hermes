"""
Agent harness — the loop. Tools and their registry live in tools.py.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

import tools

load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-2.5-flash"
MAX_ITERATIONS = 10


def run(user_message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a precise assistant. Use tools for math and file reads.",
        },
        {"role": "user", "content": user_message},
    ]

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
            else:
                try:
                    result = t.impl(**args)
                except Exception as e:
                    result = f"ERROR: {e}"

            print(f"  [OBSERVE] {str(result)[:120]}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            )

    return "Stopped: max iterations reached without a final answer."


if __name__ == "__main__":
    import sys

    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is (128*47)+99? Then read ./note.txt and tell me what it says."
    )
    print("USER:", q)
    print("\nANSWER:", run(q))
