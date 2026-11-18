"""LangGraph definition and the public ``run_agent`` entry point."""

from __future__ import annotations

import sys
import time
from typing import Any

from langgraph.graph import END, StateGraph

from agent.nodes import (
    compress_context_node,
    execute_tools_node,
    finalize_node,
    reason_node,
    route_after_reason,
    skill_select_node,
)
from agent.state import AgentState


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("skill_select", skill_select_node)
    workflow.add_node("reason", reason_node)
    workflow.add_node("tools", execute_tools_node)
    workflow.add_node("compress", compress_context_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("skill_select")
    workflow.add_edge("skill_select", "reason")
    workflow.add_conditional_edges(
        "reason",
        route_after_reason,
        {
            "tools": "tools",
            "reason": "reason",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("tools", "compress")
    workflow.add_edge("compress", "reason")
    workflow.add_edge("finalize", END)

    return workflow.compile()


compiled_graph = build_graph()


def initial_state(
    question: str,
    max_iterations: int | None = None,
    max_tool_calls: int | None = None,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    import config

    return {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "trajectory": [],
        "iterations": 0,
        "tool_calls_count": 0,
        "total_tokens": 0,
        "max_iterations": int(max_iterations or config.MAX_ITERATIONS),
        "max_tool_calls": int(max_tool_calls or config.MAX_TOOL_CALLS),
        "last_llm_latency": 0.0,
        "evidence_pool": [],
        "visited_docs": [],
        "context_summary": "",
        "candidate_answer": "",
        "skill_selected": False,
        "active_skills": [],
        "skill_context": "",
        "answer": "",
        "error": "",
    }


def run_agent(
    question: str,
    verbose: bool = False,
    max_iterations: int | None = None,
    max_tool_calls: int | None = None,
) -> dict[str, Any]:
    """Run one question through the graph and return a submission-ready record.

    ``max_iterations`` / ``max_tool_calls`` let the UI offer a fast mode without
    changing the submission defaults used by run_test.py.
    """
    question = (question or "").strip()
    started = time.time()
    try:
        final_state = compiled_graph.invoke(
            initial_state(question, max_iterations, max_tool_calls),
            config={"recursion_limit": 100},
        )
    except Exception as exc:  # pragma: no cover - last-resort guard
        message = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"[run_agent] graph failure: {message}", file=sys.stderr)
        return {
            "id": "",
            "question": question,
            "answer": "Unknown",
            "trajectory": [
                {
                    "iteration": 0,
                    "thought": f"Graph execution failed: {message}",
                    "tool": None,
                    "tool_input": None,
                    "observation": message,
                    "event": "error",
                }
            ],
            "tool_calls": 0,
            "total_tokens": 0,
            "evidence_pool": [],
            "active_skills": [],
            "elapsed_seconds": round(time.time() - started, 2),
        }

    answer = str(final_state.get("answer") or "Unknown").strip() or "Unknown"
    result = {
        "question": question,
        "answer": answer,
        "trajectory": list(final_state.get("trajectory", [])),
        "tool_calls": int(final_state.get("tool_calls_count", 0)),
        "total_tokens": int(final_state.get("total_tokens", 0)),
        "evidence_pool": list(final_state.get("evidence_pool", [])),
        "active_skills": list(final_state.get("active_skills", [])),
        "iterations": int(final_state.get("iterations", 0)),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    if verbose:
        print(
            f"[run_agent] answer={answer!r} | tools={result['tool_calls']} | "
            f"tokens={result['total_tokens']} | {result['elapsed_seconds']}s",
            file=sys.stderr,
        )
    return result


__all__ = ["compiled_graph", "build_graph", "initial_state", "run_agent", "AgentState"]
