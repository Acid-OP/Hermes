"""
Layer 3 — Context engineering: deciding what the model sees each turn.

The transcript grows every turn; it must fit the window, and reasoning degrades
when the window is bloated. So we (a) measure how full it is, (b) keep the
cacheable prefix stable, and (c) compact when it gets large. This module owns
that machinery; the loop calls into it.
"""

from __future__ import annotations


def estimate_tokens(messages) -> int:
    # Rough heuristic (~4 chars/token). Precise enough to decide WHEN to act,
    # which is all the loop needs.
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(p)) for p in content)
        for tc in m.get("tool_calls") or []:
            total += len(str(tc))
    return total // 4


def build_system_prompt(stable: str, project_context: str = "") -> str:
    # Stable tier (identity + tool guidance) and project tier change rarely, so
    # they live in the frozen prefix that providers can cache.
    prompt = stable
    if project_context:
        prompt += "\n\n# Project context\n" + project_context
    return prompt


def assemble(system_prompt: str, history: list, volatile: str = "") -> list:
    # Build the actual request without mutating stored history. The system
    # prompt stays byte-identical across turns (cache-friendly); anything
    # volatile (time, fresh memory) is appended to the LATEST user turn so it
    # never disturbs the cacheable prefix.
    msgs = [{"role": "system", "content": system_prompt}] + [dict(m) for m in history]
    if volatile:
        for m in reversed(msgs):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = m["content"] + "\n\n[context]\n" + volatile
                break
    return msgs
