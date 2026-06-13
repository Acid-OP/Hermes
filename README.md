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
- [ ] Layer 6 — Multi-agent (orchestrator-worker, subagents)
- [ ] Layer 7 — Reliability & evals (provider abstraction, rotation, guardrails)
- [ ] Layer 8 — Self-improvement & meta-loops (curation, feedback, proactive)

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install openai python-dotenv
echo "GEMINI_API_KEY=<your key>" > .env
.venv/bin/python agent.py "What is (128*47)+99? Then read ./note.txt."
```
