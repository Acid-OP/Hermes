# Hermes (build-along)

A production-grade AI agent **harness**, built from scratch one layer at a time
to internalize agent/harness engineering — not via a framework.

Reference systems studied while building this: Nous Research's Hermes Agent,
OpenClaw, and lessons from OpenAI/Anthropic harness-engineering writing.

## Layers

- [x] **Layer 1 — The Loop.** Stateless loop over an accumulating `messages`
  transcript: reason → act → observe → repeat → stop. Three stop conditions.
  Tool errors fed back as observations. Wired to Gemini via its
  OpenAI-compatible endpoint. → `agent.py`
- [x] **Layer 2 — Tools.** First-class tool registry + data/action/orchestration
  taxonomy; action tools gated by human approval + an idempotency ledger;
  large tool outputs offloaded to a store with only a reference in context.
  → `tools.py`, `agent.py`
- [x] **Layer 3 — Context engineering.** Token monitoring, frozen cacheable
  system prompt with volatile injected at the tail, cheap dedup pruning, then
  threshold-triggered middle summarization with head/tail protection, iterative
  summaries, and a deterministic fallback. → `context.py`
- [x] **Layer 4 — Memory.** SQLite durable store: episodic turns, session
  resume via recalled digest, keyword semantic search (memory_search tool),
  durable facts (remember tool), and a prefetch/write lifecycle in the loop.
  → `memory.py`
- [x] **Layer 5 — The harness.** Repo-as-system-of-record (AGENTS.md), JSONL
  run trace (observability), a task ledger (bounded scope + goal predicate),
  verification before accepting an answer, and a clean progress handoff each
  session. → `harness.py`
- [x] **Layer 6 — Multi-agent.** Subagents with isolated context + restricted
  toolsets (zero-context-cost delegation), an orchestrator-worker decompose/
  gather, an adversarial critic panel, and parallel execution. → `subagents.py`
- [x] **Layer 7 — Reliability & evals.** Provider abstraction (swap via env),
  retry with jittered backoff, error taxonomy → recovery action, a no-progress
  loop guard, and a golden-task eval harness. → `providers.py`, `llm.py`, `evals.py`
- [x] **Layer 8 — Self-improvement & proactive (capstone).** The agent writes,
  verifies, registers, and reuses its own tools (`skills.py`); a learning curve
  measures its score climbing as skills accrue (`evals.learning_curve`); and a
  proactive layer detects what's missing without being asked (`proactive.py`).

## Capstone: the self-extending agent

The headline capability is in `skills.extend()`: the agent hits something it
can't do, **writes a tool for it, tests the tool, admits it only if it passes,
registers it into the live tool registry, and reuses it forever** — bootstrapping
its own capability with zero new human code. `evals.learning_curve()` makes that
measurable: success rate climbing across rounds as skills accumulate.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install openai python-dotenv
echo "GEMINI_API_KEY=<your key>" > .env
.venv/bin/python agent.py "What is (128*47)+99? Then read ./note.txt."
```
