"""
Layer 5 — The harness: the system around the loop that makes it reliable.

Repo/workspace as the system of record (instructions live in files, not code),
an init phase, a task ledger to bound scope, verification so "done" is earned
not trusted, observability, and a clean handoff every session.
"""

from __future__ import annotations

import json
import os
import time

TRACE_PATH = os.environ.get("HARNESS_TRACE", "harness_trace.jsonl")


def trace(event: str, **data) -> None:
    # Observability is not optional for systems that act on their own: every
    # turn, tool call, and decision is appended as a structured line so a run
    # can be inspected and replayed when it goes wrong.
    rec = {"ts": time.time(), "event": event}
    rec.update(data)
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


class LoopGuard:
    # Stop condition #4: detect the agent repeating the SAME call with the same
    # args and getting nowhere (the "broken tool called 400 times" failure).
    def __init__(self, limit: int = 3):
        self.counts = {}
        self.limit = limit

    def ok(self, name: str, args) -> bool:
        key = f"{name}:{args}"
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key] < self.limit


def write_progress(session: str, status: str, summary: str = "") -> str:
    # Every session must leave a clean, readable state so the next run (or a
    # human) can pick up without guessing. The handoff artifact, not buried logs.
    path = f"harness_progress_{session}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Session: {session}\n\nstatus: {status}\n\n{summary}\n")
    return path


def verify(goal: str, answer: str):
    # "Terminal message != goal complete." Before accepting the model's answer,
    # an independent judge checks it against the goal. Returns (ok, verdict).
    from . import llm

    resp = llm.complete(
        [
            {
                "role": "system",
                "content": "You are a strict verifier. Decide if the ANSWER fully satisfies the GOAL. Reply exactly 'PASS' or 'FAIL: <one-line reason>'.",
            },
            {"role": "user", "content": f"GOAL:\n{goal}\n\nANSWER:\n{answer}"},
        ]
    )
    out = (resp.choices[0].message.content or "").strip()
    return out.upper().startswith("PASS"), out


class TaskLedger:
    # A structured task list the agent works against, instead of an open-ended
    # "do stuff" goal. Bounds scope (it can't wander) and gives an objective
    # completion predicate (all_done) that verification can check against —
    # rather than trusting the model to say it's finished.
    def __init__(self, items=None):
        self.items = [{"desc": d, "done": False} for d in (items or [])]

    def add(self, desc: str) -> None:
        self.items.append({"desc": desc, "done": False})

    def complete(self, index: int) -> None:
        if 0 <= index < len(self.items):
            self.items[index]["done"] = True

    def pending(self) -> list:
        return [i["desc"] for i in self.items if not i["done"]]

    def all_done(self) -> bool:
        return bool(self.items) and all(i["done"] for i in self.items)

    def render(self) -> str:
        return "\n".join(
            f"[{'x' if i['done'] else ' '}] {i['desc']}" for i in self.items
        )


def load_instructions(workspace: str = ".") -> str:
    # The repo is the system of record: project rules live in a file the agent
    # reads, loaded into the (cacheable) project tier of the system prompt.
    for name in ("AGENTS.md", "HARNESS.md", "CLAUDE.md"):
        path = os.path.join(workspace, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()[:4000]
    return ""
