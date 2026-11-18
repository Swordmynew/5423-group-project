"""Single source of truth for all mutable agent state."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes.

    LangGraph replaces each field with the value returned by a node, so nodes
    always return the complete updated list/dict for the fields they touch.
    """

    question: str
    messages: list[BaseMessage]

    #: Chronological, machine-readable record embedded in test_prediction.jsonl.
    trajectory: list[dict[str, Any]]

    iterations: int
    tool_calls_count: int
    total_tokens: int
    max_iterations: int
    max_tool_calls: int
    last_llm_latency: float

    #: Structured context management (Feature B).
    evidence_pool: list[dict[str, Any]]
    visited_docs: list[str]
    context_summary: str
    candidate_answer: str

    #: Skill routing / runtime loading.
    skill_selected: bool
    active_skills: list[str]
    skill_context: str

    answer: str
    error: str
