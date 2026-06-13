"""
Layer 3 — Context engineering: deciding what the model sees each turn.

The transcript grows every turn; it must fit the window, and reasoning degrades
when the window is bloated. So we (a) measure how full it is, (b) keep the
cacheable prefix stable, and (c) compact when it gets large. This module owns
that machinery; the loop calls into it.
"""

from __future__ import annotations


def estimate_tokens(messages) -> int:
    # Rough heuristic (~4 chars/token). Precise enough to decide WHEN to act,
    # which is all the loop needs.
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(p)) for p in content)
        for tc in m.get("tool_calls") or []:
            total += len(str(tc))
    return total // 4


def prune(history: list):
    # Cheap compaction, no LLM call: when the same tool output appears multiple
    # times (the same file read repeatedly), keep only the latest full copy and
    # stub the earlier ones. Reclaims tokens for free before paying for an LLM
    # summary. Returns (new_history, chars_reclaimed). Does not mutate input.
    last_index = {}
    for i, m in enumerate(history):
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            last_index[m["content"]] = i
    reclaimed = 0
    out = []
    for i, m in enumerate(history):
        is_stale_dup = (
            m.get("role") == "tool"
            and isinstance(m.get("content"), str)
            and last_index.get(m["content"]) != i
        )
        if is_stale_dup:
            reclaimed += len(m["content"])
            nm = dict(m)
            nm["content"] = "[duplicate tool output omitted — superseded by a later identical result]"
            out.append(nm)
        else:
            out.append(dict(m))
    return out, reclaimed


def compact(history: list, summarize_fn, threshold_tokens: int = 3000,
            protect_head: int = 2, protect_tail: int = 6):
    # Compact only when over threshold. Order matters: prune cheaply first, and
    # only if still large, pay an LLM to summarize the MIDDLE — protecting the
    # opening turns (original intent) and the recent turns (live context).
    history, _ = prune(history)
    if estimate_tokens(history) <= threshold_tokens:
        return history, False
    if len(history) <= protect_head + protect_tail:
        return history, False

    head = history[:protect_head]
    middle = history[protect_head:-protect_tail]
    tail = history[-protect_tail:]

    # Structural safety: never leave an assistant tool-call request in the head
    # without its results, and never let the tail begin with an orphaned tool
    # result. Both would make the assembled request invalid.
    while head and head[-1].get("tool_calls"):
        middle = [head[-1]] + middle
        head = head[:-1]
    while middle and tail and tail[0].get("role") == "tool":
        tail = [middle[-1]] + tail
        middle = middle[:-1]

    if not middle:
        return history, False

    # Iterative summaries: pull the previous summary out of the middle and feed
    # it forward, so each compaction UPDATES one running narrative instead of
    # producing disconnected fragments.
    prev_summary = ""
    fresh_middle = []
    for m in middle:
        c = m.get("content")
        if isinstance(c, str) and c.startswith("[summary of earlier turns]"):
            prev_summary = c
        else:
            fresh_middle.append(m)

    try:
        summary = summarize_fn(fresh_middle, prev_summary)
    except Exception:
        # Never lose history to a summarizer outage: fall back to a crude,
        # LLM-free summary that still keeps the continuity anchors.
        summary = _fallback_summary(fresh_middle, prev_summary)

    summary_msg = {"role": "user", "content": f"[summary of earlier turns]\n{summary}"}
    return head + [summary_msg] + tail, True


def _fallback_summary(middle: list, prev_summary: str = "") -> str:
    lines = []
    if prev_summary:
        lines.append(prev_summary.replace("[summary of earlier turns]\n", "").strip())
    for m in middle:
        role = m.get("role")
        content = str(m.get("content") or "")[:160]
        if role in ("user", "assistant", "tool") and content:
            lines.append(f"- {role}: {content}")
    return "\n".join(lines)[:2000]


def build_system_prompt(stable: str, project_context: str = "") -> str:
    # Stable tier (identity + tool guidance) and project tier change rarely, so
    # they live in the frozen prefix that providers can cache.
    prompt = stable
    if project_context:
        prompt += "\n\n# Project context\n" + project_context
    return prompt


def assemble(system_prompt: str, history: list, volatile: str = "") -> list:
    # Build the actual request without mutating stored history. The system
    # prompt stays byte-identical across turns (cache-friendly); anything
    # volatile (time, fresh memory) is appended to the LATEST user turn so it
    # never disturbs the cacheable prefix.
    msgs = [{"role": "system", "content": system_prompt}] + [dict(m) for m in history]
    if volatile:
        for m in reversed(msgs):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = m["content"] + "\n\n[context]\n" + volatile
                break
    return msgs
