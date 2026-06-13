"""
Layer 8 — Self-improvement: the agent writes, tests, curates, and reuses its
own tools. Capability is not fixed; it bootstraps. A "skill" is a synthesized
tool (a Python function + metadata) that, once verified, is registered into the
live tool registry and reused for the rest of the agent's life.

This is procedural memory: not "what happened" (episodic) but "how to do X".
"""

from __future__ import annotations

import json
import os

SKILLS_DIR = os.environ.get("HARNESS_SKILLS", "skills_store")


def _ensure():
    os.makedirs(SKILLS_DIR, exist_ok=True)


def save_skill(name: str, code: str, description: str) -> None:
    _ensure()
    with open(os.path.join(SKILLS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "code": code, "description": description}, f)


def load_skill(name: str):
    path = os.path.join(SKILLS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_skills() -> list:
    _ensure()
    out = []
    for fn in os.listdir(SKILLS_DIR):
        if fn.endswith(".json"):
            with open(os.path.join(SKILLS_DIR, fn), encoding="utf-8") as f:
                out.append(json.load(f))
    return out
