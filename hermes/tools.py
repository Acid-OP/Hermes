"""
Layer 2 — Tools.

A tool is three things, not one: the function that runs, the schema the model
sees (which is what actually steers tool choice), and a CATEGORY that tells the
harness how to treat it. Tools register themselves here; the loop dispatches
through this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from . import memory


class ToolCategory(str, Enum):
    # The taxonomy that drives safety policy downstream.
    DATA = "data"  # reads; safe, freely retryable
    ACTION = "action"  # side effects; need approval + idempotency
    ORCHESTRATION = "orchestration"  # invoke other agents (Layer 6)


@dataclass
class Tool:
    name: str
    description: str  # written FOR THE MODEL; this is what steers selection
    parameters: dict  # JSON schema of the arguments
    impl: Callable
    category: ToolCategory

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_REGISTRY: dict = {}


def tool(name, description, parameters, category=ToolCategory.DATA):
    def deco(fn):
        _REGISTRY[name] = Tool(name, description, parameters, fn, category)
        return fn

    return deco


def all_schemas() -> list:
    return [t.schema() for t in _REGISTRY.values()]


def get(name):
    return _REGISTRY.get(name)


# Offload store: full tool outputs live here keyed by id; the context gets only
# a short reference. Keeps a single big tool result from poisoning every later
# turn's reasoning (and from costing tokens forever). This is the seam into
# Layer 3 (context engineering).
TOOL_LOG: dict = {}


# --------------------------------------------------------------------------
# The tools
# --------------------------------------------------------------------------


@tool(
    name="calculator",
    description="Evaluate a Python arithmetic expression like '2*(3+4)'. Use for ALL math; never compute in your head.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
    category=ToolCategory.DATA,
)
def calculator(expression: str) -> str:
    return str(eval(expression))


@tool(
    name="read_file",
    description="Read a UTF-8 text file from disk; returns up to 2000 chars.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    category=ToolCategory.DATA,
)
def read_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()[:2000]
    except Exception as e:
        return f"ERROR: {e}"


@tool(
    name="remember",
    description="Persist a durable fact as key/value for future sessions (a preference, a decision, an entity).",
    parameters={
        "type": "object",
        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
        "required": ["key", "value"],
    },
    category=ToolCategory.DATA,
)
def remember(key: str, value: str) -> str:
    memory.remember_fact(key, value)
    return f"remembered: {key}"


@tool(
    name="memory_search",
    description="Search durable memory of past turns. Use before answering anything about prior work, decisions, or facts the user gave you earlier.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    category=ToolCategory.DATA,
)
def memory_search(query: str) -> str:
    hits = memory.search(query)
    return "\n".join(hits) if hits else "(no relevant memory)"


@tool(
    name="read_tool_log",
    description="Retrieve the full content of an offloaded tool result by its log id.",
    parameters={
        "type": "object",
        "properties": {"log_id": {"type": "string"}},
        "required": ["log_id"],
    },
    category=ToolCategory.DATA,
)
def read_tool_log(log_id: str) -> str:
    return TOOL_LOG.get(log_id, f"ERROR: no such log id {log_id}")


@tool(
    name="write_file",
    description="Write text to a file on disk (overwrites). Use to save results.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    # ACTION: it has a side effect, so the harness gates it (approval + idempotency).
    category=ToolCategory.ACTION,
)
def write_file(path: str, content: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {path}"
