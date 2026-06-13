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
