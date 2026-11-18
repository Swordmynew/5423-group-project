"""Skill-augmented search agent package.

The graph is imported lazily so that ``import agent.nodes`` does not create a
circular import through the package initialiser.
"""

from typing import Any

__all__ = ["compiled_graph", "run_agent"]


def __getattr__(name: str) -> Any:
    if name in {"compiled_graph", "run_agent"}:
        from agent import graph as _graph

        return getattr(_graph, name)
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
