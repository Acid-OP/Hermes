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
