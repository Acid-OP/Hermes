"""
Layer 7 — Evals: measure whether the agent works, repeatably.

Without an eval suite you tune blind — every change is a guess. A golden set of
tasks with known answers, a runner, and a score. This is also the substrate the
Layer 8 self-improvement loop measures itself against (the learning curve).
"""

from __future__ import annotations

GOLDEN = [
    {"task": "What is 17*23? Reply with just the number.", "expect": "391"},
    {"task": "What is (100-1)*2? Reply with just the number.", "expect": "198"},
    {"task": "Reply with exactly the word: ready", "expect": "ready"},
]


def score_exact(answer, expect) -> bool:
    return str(expect).strip().lower() in str(answer).strip().lower()


def run_suite(run_fn, tasks=None) -> dict:
    tasks = tasks or GOLDEN
    results = []
    for t in tasks:
        try:
            ans = run_fn(t["task"])
        except Exception as e:
            ans = f"ERROR: {e}"
        results.append(
            {"task": t["task"], "pass": score_exact(ans, t["expect"]), "answer": str(ans)[:120]}
        )
    passed = sum(1 for r in results if r["pass"])
    return {
        "passed": passed,
        "total": len(results),
        "score": passed / len(results) if results else 0.0,
        "results": results,
    }
