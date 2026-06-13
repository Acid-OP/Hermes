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
