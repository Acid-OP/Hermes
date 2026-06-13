"""
Layer 5 — The harness: the system around the loop that makes it reliable.

Repo/workspace as the system of record (instructions live in files, not code),
an init phase, a task ledger to bound scope, verification so "done" is earned
not trusted, observability, and a clean handoff every session.
"""

from __future__ import annotations

import os


def load_instructions(workspace: str = ".") -> str:
    # The repo is the system of record: project rules live in a file the agent
    # reads, loaded into the (cacheable) project tier of the system prompt.
    for name in ("AGENTS.md", "HARNESS.md", "CLAUDE.md"):
        path = os.path.join(workspace, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()[:4000]
    return ""
