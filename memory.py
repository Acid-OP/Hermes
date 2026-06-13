"""
Layer 4 — Memory: what persists beyond a single run.

The live transcript (Layer 1) is working memory and dies with the run. This is
durable memory: episodic turns saved across sessions, keyword recall over them,
and explicit facts. Memory is for RECALL, not exact replay — we store text, not
the full tool-call structure, and inject relevant pieces back into context.
SQLite is the single backend (episodic + semantic + facts in one engine).
"""

from __future__ import annotations

import os
import sqlite3
import time

DB_PATH = os.environ.get("HARNESS_DB", "harness_memory.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute(
        "CREATE TABLE IF NOT EXISTS turns "
        "(id INTEGER PRIMARY KEY, session TEXT, role TEXT, content TEXT, ts REAL)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS facts "
        "(id INTEGER PRIMARY KEY, key TEXT UNIQUE, value TEXT, ts REAL)"
    )
    return c


def save_turn(session: str, role: str, content: str) -> None:
    c = _conn()
    c.execute(
        "INSERT INTO turns(session, role, content, ts) VALUES(?,?,?,?)",
        (session, role, str(content), time.time()),
    )
    c.commit()
    c.close()


def load_session(session: str) -> list:
    c = _conn()
    rows = c.execute(
        "SELECT role, content FROM turns WHERE session=? ORDER BY id", (session,)
    ).fetchall()
    c.close()
    return [{"role": r, "content": t} for r, t in rows]


def remember_fact(key: str, value: str) -> None:
    # Durable facts (preferences, decisions, entities) keyed for easy recall —
    # the user-model / entity slice of memory. Upsert keeps one value per key.
    c = _conn()
    c.execute(
        "INSERT OR REPLACE INTO facts(key, value, ts) VALUES(?,?,?)",
        (key, value, time.time()),
    )
    c.commit()
    c.close()


def recall_facts() -> dict:
    c = _conn()
    rows = c.execute("SELECT key, value FROM facts ORDER BY ts DESC").fetchall()
    c.close()
    return {k: v for k, v in rows}


def search(query: str, limit: int = 5) -> list:
    # Keyword recall across all stored turns (the FTS-style approach — Hermes's
    # lane). A real system swaps this for embedding/vector search; the interface
    # stays the same. Ranked by term frequency.
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        return []
    c = _conn()
    rows = c.execute("SELECT role, content FROM turns").fetchall()
    c.close()
    scored = []
    for role, content in rows:
        score = sum(content.lower().count(t) for t in terms)
        if score:
            scored.append((score, role, content))
    scored.sort(reverse=True)
    return [f"{r}: {c[:200]}" for _, r, c in scored[:limit]]


def session_digest(session: str, limit: int = 20) -> str:
    # A compact recall of recent prior turns, to inject as context on resume.
    rows = load_session(session)[-limit:]
    if not rows:
        return ""
    return "\n".join(f"{r['role']}: {r['content'][:200]}" for r in rows)
