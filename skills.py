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
import re

SKILLS_DIR = os.environ.get("HARNESS_SKILLS", "skills_store")


def _parse_spec(text: str) -> dict:
    # Models wrap JSON in ``` fences; strip them, then parse the spec.
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    return json.loads(t)


def synthesize(capability: str) -> dict:
    # The agent writes a new tool. Returns a spec {name, description, code}.
    import llm

    resp = llm.complete(
        [
            {
                "role": "system",
                "content": 'Write ONE Python function implementing the capability. Reply ONLY with JSON: {"name": str, "description": str, "code": str}. The function takes simple string/number args and returns a string. Use stdlib only.',
            },
            {"role": "user", "content": capability},
        ]
    )
    return _parse_spec(resp.choices[0].message.content)


def _ensure():
    os.makedirs(SKILLS_DIR, exist_ok=True)


def compile_skill(spec: dict):
    # Turn skill code into a callable. exec of model-written code is unsafe —
    # a lab shortcut; in production this runs in a sandbox (your Lovable lesson).
    ns: dict = {}
    exec(spec["code"], ns)
    fn = ns.get(spec["name"])
    if not callable(fn):
        raise ValueError(f"skill code did not define {spec['name']}")
    return fn


def verify_skill(spec: dict, test_input, expected) -> bool:
    # Never admit generated code blindly: it must pass a test first. This is the
    # verification discipline turned on the agent's own output.
    try:
        fn = compile_skill(spec)
        out = fn(**test_input) if isinstance(test_input, dict) else fn(test_input)
        return str(expected) in str(out)
    except Exception:
        return False


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
