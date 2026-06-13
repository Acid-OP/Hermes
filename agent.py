"""
Agent harness — the loop. Model access is in llm.py, tools in tools.py,
context machinery in context.py.
"""

import hashlib
import json
import sys

import context
import harness
import llm
import memory
import tools
from tools import ToolCategory

MAX_ITERATIONS = 10

# Stable tier of the system prompt — frozen for the whole session so the
# provider can cache this prefix across turns.
SYSTEM_PROMPT = "You are a precise assistant. Use tools for math, file reads, and file writes."


def _idempotency_key(name, args) -> str:
    return hashlib.sha256(
        (name + json.dumps(args, sort_keys=True)).encode()
    ).hexdigest()[:12]


def _approve(tool, args) -> bool:
    if not sys.stdin.isatty():
        print(f"  [APPROVE] auto-approving '{tool.name}' (non-interactive run)")
        return True
    return input(f"  [APPROVE] run {tool.name}({args})? [y/N] ").strip().lower() == "y"


COMPACTION_THRESHOLD = 3000  # tokens; compact the transcript past this


def _summarize(middle: list, prev_summary: str = "") -> str:
    text = "\n".join(
        f"{m.get('role')}: {str(m.get('content'))[:500]}" for m in middle
    )
    instruction = "Summarize these agent turns concisely. Preserve decisions, results, file names, and errors."
    if prev_summary:
        instruction = (
            "Update the running summary below to incorporate the new turns, "
            "preserving continuity. Keep it concise.\n\n" + prev_summary
        )
    resp = llm.complete(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ]
    )
    return resp.choices[0].message.content


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


def run(user_message: str, session_id: str = "default") -> str:
    # Durable memory: recall prior turns from this session and persist new ones.
    memory.save_turn(session_id, "user", user_message)
    digest = memory.session_digest(session_id)
    instructions = harness.load_instructions()
    project_context = "\n\n".join(p for p in [instructions, digest] if p)
    system_prompt = context.build_system_prompt(SYSTEM_PROMPT, project_context=project_context)

    # Prefetch: pull durable facts + memory relevant to this request and inject
    # them as volatile context (read-before-turn). Writes happen during the run
    # (remember tool) and after (save_turn) — the full read/write lifecycle.
    facts = memory.recall_facts()
    prefetch = memory.search(user_message)
    volatile_parts = []
    if facts:
        volatile_parts.append("known facts: " + "; ".join(f"{k}={v}" for k, v in facts.items()))
    if prefetch:
        volatile_parts.append("relevant memory:\n" + "\n".join(prefetch))
    volatile = "\n".join(volatile_parts)

    # history holds only the conversation (no system prompt). The system prompt
    # is assembled fresh each turn and kept frozen, so it stays cacheable.
    history = [{"role": "user", "content": user_message}]
    done_actions: dict = {}
    tools.TOOL_LOG.clear()

    for iteration in range(MAX_ITERATIONS):
        history, compacted = context.compact(history, _summarize, COMPACTION_THRESHOLD)
        if compacted:
            print("  [COMPACT] summarized middle of transcript")
        print(f"\n--- iteration {iteration + 1} (~{context.estimate_tokens(history)} tok) ---")
        api_messages = context.assemble(system_prompt, history, volatile=volatile)
        resp = llm.complete(api_messages, tools=tools.all_schemas())
        msg = resp.choices[0].message
        history.append(msg.model_dump(exclude_none=True))

        harness.trace("turn", session=session_id, iteration=iteration + 1,
                      tokens=context.estimate_tokens(history),
                      tool_calls=[tc.function.name for tc in (msg.tool_calls or [])])

        if not msg.tool_calls:
            memory.save_turn(session_id, "assistant", msg.content)
            harness.trace("final", session=session_id, answer=str(msg.content)[:300])
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
            harness.trace("tool", session=session_id, name=name, args=args,
                          result=str(result)[:200])
            print(f"  [OBSERVE] {str(result)[:120]}")
            history.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            )

    return "Stopped: max iterations reached without a final answer."


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is (128*47)+99? Then read ./note.txt."
    print("USER:", q)
    print("\nANSWER:", run(q))
