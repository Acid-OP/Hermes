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


def learning_curve(run_fn, improve_fn=None, rounds: int = 3, tasks=None) -> list:
    # The self-improvement demo: run the suite repeatedly; between rounds the
    # agent improves itself on the tasks it failed (improve_fn). A self-extending
    # agent's score should CLIMB across rounds. Returns the score per round.
    curve = []
    for _ in range(rounds):
        res = run_suite(run_fn, tasks)
        curve.append(res["score"])
        if improve_fn:
            failures = [r["task"] for r in res["results"] if not r["pass"]]
            if failures:
                improve_fn(failures)
    return curve


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
