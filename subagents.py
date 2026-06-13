"""
Layer 6 — Multi-agent: delegation, orchestration, adversarial verification.

A subagent runs its OWN loop with an isolated context (a fresh transcript, no
view of the parent's history) and its own budget, and returns only a distilled
result. That isolation is what makes delegation "zero-context-cost": the
parent's window stays clean no matter how many steps the child took.
"""

from __future__ import annotations

import json

import llm
import tools
from tools import ToolCategory


def _select_schemas(allowed):
    # Subagents get a restricted toolset. By default only DATA tools, so a child
    # can read/think but cannot fire side effects (no approval human is present).
    if allowed is None:
        allowed = [
            t.schema()["function"]["name"]
            for t in tools._REGISTRY.values()
            if t.category == ToolCategory.DATA
        ]
    return [s for s in tools.all_schemas() if s["function"]["name"] in allowed]


def decompose(task: str) -> list:
    resp = llm.complete(
        [
            {"role": "system", "content": "Break the task into 2-4 independent subtasks. Reply as a JSON array of strings, nothing else."},
            {"role": "user", "content": task},
        ]
    )
    try:
        subs = json.loads(resp.choices[0].message.content)
        if isinstance(subs, list) and subs:
            return [str(s) for s in subs]
    except Exception:
        pass
    return [task]  # fall back to treating it as one task


def orchestrate(task: str) -> dict:
    # Orchestrator-worker: a lead decomposes, isolated workers each solve a
    # piece, results are gathered. The parent never sees the workers' steps.
    subtasks = decompose(task)
    results = [spawn(s) for s in subtasks]
    return {"subtasks": subtasks, "results": results}


def _tally(verdicts) -> bool:
    # Survives only if fewer than half the skeptics refuted it.
    n = len(verdicts)
    refuted = sum(1 for v in verdicts if not str(v).strip().upper().startswith("SOUND"))
    return refuted < (n + 1) // 2


def critique(claim: str, n: int = 3) -> dict:
    # Adversarial verification: N independent skeptics each TRY to refute the
    # claim (no tools, pure reasoning). Majority refutation kills it. Diverse
    # attacks catch failure modes a single check would miss.
    verdicts = [
        spawn(
            f"Try to find a flaw or counterexample. If the claim is sound reply exactly 'SOUND', else 'FLAW: <reason>'.\n\nCLAIM: {claim}",
            max_iterations=2,
            allowed=[],
        )
        for _ in range(n)
    ]
    return {"survives": _tally(verdicts), "verdicts": verdicts}


def spawn(task: str, max_iterations: int = 6, allowed=None) -> str:
    history = [
        {"role": "system", "content": "You are a focused subagent. Do exactly the task and report the result concisely."},
        {"role": "user", "content": task},
    ]
    schemas = _select_schemas(allowed)
    for _ in range(max_iterations):
        resp = llm.complete(history, tools=schemas)
        msg = resp.choices[0].message
        history.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content  # only the distilled result crosses back to the parent
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            t = tools.get(name)
            result = t.impl(**args) if t else f"ERROR: unknown tool {name}"
            history.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
    return "(subagent hit its iteration cap)"


@tools.tool(
    name="delegate",
    description="Delegate a self-contained subtask to an isolated subagent; returns only its distilled result.",
    parameters={
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    },
    category=ToolCategory.ORCHESTRATION,
)
def delegate(task: str) -> str:
    return spawn(task)
