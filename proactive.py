"""
Layer 8 — Proactive: don't wait to be asked.

The difference between a reactive tool and a real agent is that the agent
notices what is MISSING. Gap detection is deterministic (a goal-state checklist
vs. the current state) and runs on its own; acting on the gaps is then agentic.
This is the Ostrya thesis: "you created your course — these 3 things are still
missing."
"""

from __future__ import annotations


def detect_gaps(required: list, present: list) -> list:
    # Programmatic, always-on: what does a complete X need that isn't there yet.
    present_set = {p.lower().strip() for p in present}
    return [r for r in required if r.lower().strip() not in present_set]


def proactive_message(domain: str, gaps: list) -> str:
    if not gaps:
        return f"Your {domain} looks complete."
    return (
        f"Your {domain} is still missing: "
        + "; ".join(gaps)
        + ". Want me to set these up next?"
    )
