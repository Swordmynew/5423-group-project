"""LangGraph nodes: skill routing, reasoning, tool execution, and context control."""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt import ToolNode

import config
from agent.skill_loader import list_available_skills, load_skill, skill_catalog_text
from agent.state import AgentState
from index.text_utils import clean_final_answer, make_snippet
from tools import TOOLS

# ---------------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------------

_LLM = None
_LLM_WITH_TOOLS = None
_LLM_LOCK = threading.RLock()
_TOOL_NODE: ToolNode | None = None


def get_llm():
    """Return the shared ChatOpenAI client for DeepSeek-V4-Flash."""
    global _LLM
    if _LLM is None:
        with _LLM_LOCK:
            if _LLM is None:
                try:
                    from langchain_openai import ChatOpenAI
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "langchain-openai is required. Install it with `pip install langchain-openai`."
                    ) from exc
                _LLM = ChatOpenAI(
                    model=config.DEEPSEEK_MODEL,
                    api_key=config.require_api_key(),
                    base_url=config.DEEPSEEK_BASE_URL,
                    temperature=config.LLM_TEMPERATURE,
                    timeout=config.LLM_TIMEOUT_SECONDS,
                    max_retries=config.LLM_MAX_RETRIES,
                    max_tokens=config.LLM_MAX_TOKENS,
                )
    return _LLM


def get_llm_with_tools():
    global _LLM_WITH_TOOLS
    if _LLM_WITH_TOOLS is None:
        with _LLM_LOCK:
            if _LLM_WITH_TOOLS is None:
                _LLM_WITH_TOOLS = get_llm().bind_tools(TOOLS)
    return _LLM_WITH_TOOLS


def get_tool_node() -> ToolNode:
    global _TOOL_NODE
    if _TOOL_NODE is None:
        with _LLM_LOCK:
            if _TOOL_NODE is None:
                try:
                    _TOOL_NODE = ToolNode(TOOLS, handle_tool_error=True)
                except TypeError:
                    try:
                        # LangGraph 0.3 uses the plural spelling.
                        _TOOL_NODE = ToolNode(TOOLS, handle_tool_errors=True)
                    except TypeError:  # very old signature
                        _TOOL_NODE = ToolNode(TOOLS)
    return _TOOL_NODE


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in {"text", "output_text", None} and block.get("text"):
                    parts.append(str(block["text"]))
                elif block.get("content"):
                    parts.append(_content_text(block["content"]))
        return "".join(parts)
    return str(content)


def _message_text(message: BaseMessage | None) -> str:
    if message is None:
        return ""
    return _content_text(getattr(message, "content", ""))


def _usage_tokens(response: Any) -> int:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage.get("total_tokens") is not None:
        return int(usage["total_tokens"])
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if isinstance(token_usage, dict):
        return int(token_usage.get("total_tokens", 0) or 0)
    return 0


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"FINAL\s+ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    answer = match.group(1).strip()
    answer = answer.splitlines()[0].strip()
    answer = re.sub(r"^[\"'`*_]+|[\"'`*_]+$", "", answer).strip()
    return answer


def _has_final_marker(message: BaseMessage | None) -> bool:
    return bool(re.search(r"FINAL\s+ANSWER\s*:", _message_text(message), flags=re.IGNORECASE))


def _tool_call_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or "")
    return str(getattr(call, "name", ""))


def _tool_call_args(call: Any) -> dict:
    if isinstance(call, dict):
        args = call.get("args") or {}
    else:
        args = getattr(call, "args", {}) or {}
    return args if isinstance(args, dict) else {"input": args}


def _tool_call_id(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("id") or "")
    return str(getattr(call, "id", "") or "")


def _estimate_tokens(messages: list[BaseMessage]) -> int:
    total_chars = 0
    for message in messages:
        total_chars += len(_message_text(message))
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=False, default=str))
    return total_chars // 4 + 8 * len(messages)


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """You are a research agent answering a difficult question over a fixed corpus of 100,195 web documents. Retrieval returns whole documents; snippets are only 160 characters, so open a promising document before trusting a detail.

Strategy:
1. Search the rarest clue first (unusual name, exact phrase, date, number, relation), not the whole question.
2. Use document_reader with focus=... on the best candidate; use text_searcher inside a document for an exact phrase.
3. Cross-check the final candidate against a document that explicitly supports it.
4. If results are poor, rewrite the query, use a synonym, drop one constraint, or try metadata_filter.
5. Stop within about 8 tool calls once the evidence is sufficient; prefer one tool call per turn.

Rules:
- Never invent a document ID or a fact.
- Prefer short specific queries (2-8 words).
- Several independent tool calls may be issued in one turn.
- There is no live web search.

Finish with exactly one line:
FINAL ANSWER: <short answer: name, entity, date, or number>

Do not add explanations after the FINAL ANSWER line.

If evidence is genuinely insufficient, commit to the single most likely answer after checking the best candidates."""


def _format_evidence_pool(pool: list[dict[str, Any]]) -> str:
    if not pool:
        return "EVIDENCE POOL: empty."
    lines = ["EVIDENCE POOL (highest relevance first):"]
    for entry in pool[: config.EVIDENCE_POOL_MAX]:
        doc_id = entry.get("doc_id", "?")
        snippet = make_snippet(entry.get("snippet", ""), "", 150)
        lines.append(f"- doc_id={doc_id} | {snippet}")
    return "\n".join(lines)


def build_system_prompt(state: AgentState, nudge: bool = False) -> str:
    parts = [BASE_SYSTEM_PROMPT]

    parts.append(f"\nQUESTION:\n{state.get('question', '')}")

    if state.get("skill_context"):
        names = ", ".join(state.get("active_skills", []))
        parts.append(f"\nACTIVE SKILL(S): {names}\n{state['skill_context']}")

    parts.append("\n" + _format_evidence_pool(state.get("evidence_pool", [])))

    iterations = int(state.get("iterations", 0))
    max_iterations = int(state.get("max_iterations", config.MAX_ITERATIONS))
    parts.append(
        f"\nPROGRESS: iteration {iterations}/{max_iterations}; "
        f"tool calls so far {int(state.get('tool_calls_count', 0))}/"
        f"{int(state.get('max_tool_calls', config.MAX_TOOL_CALLS))}."
    )
    if state.get("candidate_answer"):
        parts.append(f"CANDIDATE ANSWER (verify before finalising): {state['candidate_answer']}")
    if iterations >= max_iterations - 1:
        parts.append("This is the final iteration: either answer now or make one decisive tool call.")

    if nudge:
        parts.append(
            "\nYour previous turn contained neither a tool call nor a FINAL ANSWER line. "
            "Take one concrete action now: call a tool, or write 'FINAL ANSWER: <answer>'."
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Skill selection node
# ---------------------------------------------------------------------------


def _parse_skill_selection(raw: str, available_names: list[str]) -> tuple[list[str], str]:
    text = raw.strip()
    data: Any = None
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None

    selected: list[str] = []
    reason = ""
    if isinstance(data, dict):
        value = data.get("skills") or data.get("skill") or []
        if isinstance(value, str):
            selected = [value]
        elif isinstance(value, list):
            selected = [str(item) for item in value]
        reason = str(data.get("reason") or "")
    elif isinstance(data, list):
        selected = [str(item) for item in data]

    if not selected:
        lowered = text.lower()
        selected = [name for name in available_names if name.lower() in lowered]

    valid: list[str] = []
    for name in selected:
        cleaned = name.strip()
        if cleaned in available_names and cleaned not in valid:
            valid.append(cleaned)
    if not reason:
        reason = text[:300]
    return valid[:2], reason


def _heuristic_skills(question: str) -> list[str]:
    lowered = question.lower()
    if "pdf" in lowered or "paper" in lowered or "report" in lowered:
        return ["pdf-extractor", "multi-clue-decomposer"]
    if len(question) > 220 or question.count("?") >= 2 or lowered.count(" and ") >= 2:
        return ["multi-clue-decomposer"]
    return []


def skill_select_node(state: AgentState) -> dict[str, Any]:
    """Choose 0-2 skills from metadata only, then load their bodies at runtime."""
    question = state.get("question", "")
    available = list_available_skills()
    available_names = [skill["name"] for skill in available]
    trajectory = list(state.get("trajectory", []))
    total_tokens = int(state.get("total_tokens", 0))

    selected: list[str] = []
    rationale = "No skills are installed."
    if available:
        prompt = (
            "You are routing a research question to optional reusable skills.\n"
            "Available skills (name: description):\n"
            f"{skill_catalog_text()}\n\n"
            f"Question:\n{question}\n\n"
            "Pick the 0-2 skills that will actually help this question. "
            "Choose none if the question is simple and a plain search is enough. "
            "Do not pick a skill merely because it exists.\n"
            'Return only JSON: {"skills": ["name"], "reason": "one sentence"}'
        )
        try:
            response = get_llm().invoke(
                [
                    SystemMessage(content="You are a concise routing module. Return valid JSON only."),
                    HumanMessage(content=prompt),
                ]
            )
            total_tokens += _usage_tokens(response)
            selected, rationale = _parse_skill_selection(
                _content_text(response.content), available_names
            )
        except Exception as exc:
            selected = _heuristic_skills(question)
            rationale = f"LLM routing unavailable ({type(exc).__name__}); used heuristic routing."

    contexts: list[str] = []
    loaded: list[str] = []
    for name in selected:
        skill = load_skill(name)
        if skill:
            loaded.append(name)
            contexts.append(
                f"### SKILL: {skill['name']}\n{skill['description']}\n\n{skill['body']}"
            )
    selected = loaded

    if selected:
        observation = f"Loaded skill(s): {', '.join(selected)}."
    else:
        observation = "No skill selected; using the base search procedure."

    trajectory.append(
        {
            "iteration": 0,
            "thought": f"Skill routing: {rationale}",
            "tool": None,
            "tool_input": None,
            "observation": observation,
            "event": "skill_selection",
        }
    )
    return {
        "skill_selected": True,
        "active_skills": selected,
        "skill_context": "\n\n".join(contexts),
        "trajectory": trajectory,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Reasoning node
# ---------------------------------------------------------------------------


def reason_node(state: AgentState) -> dict[str, Any]:
    """Ask the tool-bound LLM for the next action."""
    messages = list(state.get("messages", []))
    iterations = int(state.get("iterations", 0))
    max_iterations = int(state.get("max_iterations", config.MAX_ITERATIONS))
    if iterations >= max_iterations:
        return {}
    if int(state.get("tool_calls_count", 0)) >= int(
        state.get("max_tool_calls", config.MAX_TOOL_CALLS)
    ):
        # Hard efficiency guard: finalise with the evidence already collected.
        return {"error": "tool_call_budget_exhausted"}

    last = messages[-1] if messages else None
    nudge = bool(
        isinstance(last, AIMessage)
        and not getattr(last, "tool_calls", None)
        and not _has_final_marker(last)
    )

    prompt_messages: list[BaseMessage] = [
        SystemMessage(content=build_system_prompt(state, nudge=nudge))
    ]
    conversation = list(messages)
    # After compression the first message is the summary system message.  Merge
    # it into the main system prompt so the request has a single system block.
    if conversation and isinstance(conversation[0], SystemMessage):
        prompt_messages[0] = SystemMessage(
            content=_message_text(prompt_messages[0]) + "\n\n" + _message_text(conversation[0])
        )
        conversation = conversation[1:]
    prompt_messages.extend(conversation)

    trajectory = list(state.get("trajectory", []))
    total_tokens = int(state.get("total_tokens", 0))
    candidate_answer = state.get("candidate_answer", "")
    error = ""
    llm_started = time.time()
    llm_latency = 0.0

    try:
        response = get_llm_with_tools().invoke(prompt_messages)
        llm_latency = round(time.time() - llm_started, 2)
        total_tokens += _usage_tokens(response)
    except Exception as exc:
        llm_latency = round(time.time() - llm_started, 2)
        error = f"{type(exc).__name__}: {exc}"
        response = AIMessage(
            content=(
                "I could not reach the language model. "
                f"FINAL ANSWER: {candidate_answer or 'Unknown'}"
            )
        )
        # Do not count a failed call as a generated token event; the state will
        # finalise with the best candidate we have.
        total_tokens = int(state.get("total_tokens", 0))
        trajectory.append(
            {
                "iteration": iterations + 1,
                "thought": f"LLM error: {error}",
                "tool": None,
                "tool_input": None,
                "observation": error,
                "event": "error",
                "latency_seconds": llm_latency,
            }
        )

    content = _message_text(response)
    messages.append(response)

    extracted = _extract_final_answer(content)
    if extracted:
        candidate_answer = extracted

    if not error and not getattr(response, "tool_calls", None) and not extracted:
        trajectory.append(
            {
                "iteration": iterations + 1,
                "thought": content,
                "tool": None,
                "tool_input": None,
                "observation": content or "Model returned no content.",
                "latency_seconds": llm_latency,
            }
        )

    return {
        "messages": messages,
        "iterations": iterations + 1,
        "total_tokens": total_tokens,
        "trajectory": trajectory,
        "candidate_answer": candidate_answer,
        "error": error,
        "last_llm_latency": llm_latency,
    }


# ---------------------------------------------------------------------------
# Tool execution + evidence management
# ---------------------------------------------------------------------------


def _evidence_entries_from_tool(content: str, tool_name: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(content)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []

    entries: list[dict[str, Any]] = []
    for result in data.get("results") or []:
        if isinstance(result, dict) and result.get("doc_id"):
            entries.append(
                {
                    "doc_id": str(result["doc_id"]),
                    "snippet": make_snippet(result.get("snippet", ""), "", 150),
                    "relevance_score": float(result.get("score", 0.0) or 0.0),
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "source": tool_name,
                }
            )

    if data.get("doc_id") and ("excerpt" in data or "text" in data):
        entries.append(
            {
                "doc_id": str(data["doc_id"]),
                "snippet": make_snippet(data.get("excerpt") or data.get("text") or "", "", 150),
                "relevance_score": 0.05,
                "url": data.get("url", ""),
                "title": data.get("title", ""),
                "source": tool_name,
            }
        )

    for match in data.get("matches") or []:
        if isinstance(match, dict) and match.get("doc_id"):
            entries.append(
                {
                    "doc_id": str(match["doc_id"]),
                    "snippet": make_snippet(match.get("snippet", ""), "", 150),
                    "relevance_score": 0.04,
                    "url": match.get("url", ""),
                    "title": match.get("title", ""),
                    "source": tool_name,
                }
            )
    return entries


def update_evidence_pool(
    pool: list[dict[str, Any]],
    tool_calls: list[Any] | None = None,
    tool_messages: list[ToolMessage] | None = None,
) -> list[dict[str, Any]]:
    """Merge new tool evidence into a small, relevance-ranked pool.

    The soft cap is 5 entries.  Pruning by ``relevance_score`` is triggered when
    the pool grows beyond 8; the pool is then reduced to the best 5.
    """
    merged: dict[str, dict[str, Any]] = {
        str(entry.get("doc_id")): dict(entry)
        for entry in (pool or [])
        if entry.get("doc_id")
    }

    for message in tool_messages or []:
        name = getattr(message, "name", "") or ""
        for entry in _evidence_entries_from_tool(_content_text(message.content), name):
            doc_id = entry["doc_id"]
            current = merged.get(doc_id)
            if current is None or entry["relevance_score"] >= float(
                current.get("relevance_score", 0.0)
            ):
                merged[doc_id] = entry

    entries = [dict(entry) for entry in merged.values()]
    threshold = config.EVIDENCE_POOL_PRUNE_THRESHOLD
    maximum = config.EVIDENCE_POOL_MAX
    if len(entries) > threshold:
        entries = sorted(
            entries, key=lambda item: float(item.get("relevance_score", 0.0)), reverse=True
        )[:maximum]
    elif len(entries) > maximum:
        entries = sorted(
            entries, key=lambda item: float(item.get("relevance_score", 0.0)), reverse=True
        )[:maximum]
    return entries


def _tool_messages_from_result(result: Any) -> list[ToolMessage]:
    if isinstance(result, dict):
        raw = result.get("messages", [])
    elif isinstance(result, (list, tuple)):
        raw = list(result)
    else:
        raw = []
    return [message for message in raw if isinstance(message, ToolMessage)]


def execute_tools_node(state: AgentState) -> dict[str, Any]:
    """Execute the pending tool calls and record observations chronologically."""
    messages = list(state.get("messages", []))
    if not messages:
        return {}
    last = messages[-1]
    calls = list(getattr(last, "tool_calls", None) or [])
    if not calls:
        return {}

    try:
        tool_started = time.time()
        result = get_tool_node().invoke({"messages": [last]})
        tool_latency = round(time.time() - tool_started, 2)
        tool_messages = _tool_messages_from_result(result)
    except Exception as exc:
        tool_latency = 0.0
        tool_messages = []
        error_text = f"Tool execution failed: {type(exc).__name__}: {exc}"

    by_call_id = {str(getattr(message, "tool_call_id", "")): message for message in tool_messages}
    ordered_messages: list[ToolMessage] = []
    observations: dict[str, str] = {}
    for call in calls:
        call_id = _tool_call_id(call)
        message = by_call_id.get(call_id)
        if message is None:
            message = ToolMessage(
                content=json.dumps(
                    {"error": error_text if "error_text" in locals() else "Tool returned no result."},
                    ensure_ascii=False,
                ),
                tool_call_id=call_id,
                name=_tool_call_name(call),
            )
        ordered_messages.append(message)
        observations[call_id] = _message_text(message)

    messages.extend(ordered_messages)
    trajectory = list(state.get("trajectory", []))
    iteration = int(state.get("iterations", 0))
    thought = _message_text(last)

    visited = list(state.get("visited_docs", []))
    for call in calls:
        call_id = _tool_call_id(call)
        observation = observations.get(call_id, "")
        trajectory.append(
            {
                "iteration": iteration,
                "thought": thought,
                "tool": _tool_call_name(call),
                "tool_input": _tool_call_args(call),
                "observation": _truncate(observation, config.TRAJECTORY_OBSERVATION_MAX_CHARS),
                "latency_seconds": float(state.get("last_llm_latency", 0.0)),
                "tool_seconds": tool_latency,
            }
        )
        try:
            data = json.loads(observation)
        except Exception:
            continue
        doc_ids: list[str] = []
        if isinstance(data, dict):
            if data.get("doc_id"):
                doc_ids.append(str(data["doc_id"]))
            for item in data.get("results") or []:
                if isinstance(item, dict) and item.get("doc_id"):
                    doc_ids.append(str(item["doc_id"]))
            for item in data.get("matches") or []:
                if isinstance(item, dict) and item.get("doc_id"):
                    doc_ids.append(str(item["doc_id"]))
        for doc_id in doc_ids:
            if doc_id not in visited:
                visited.append(doc_id)

    evidence_pool = update_evidence_pool(
        state.get("evidence_pool", []), tool_calls=calls, tool_messages=ordered_messages
    )
    return {
        "messages": messages,
        "tool_calls_count": int(state.get("tool_calls_count", 0)) + len(calls),
        "trajectory": trajectory,
        "evidence_pool": evidence_pool,
        "visited_docs": visited,
    }


# ---------------------------------------------------------------------------
# Context compression node
# ---------------------------------------------------------------------------


def _compression_split(messages: list[BaseMessage], keep: int) -> int:
    """Return a safe split index so the retained suffix never starts with a ToolMessage."""
    start = max(0, len(messages) - keep)
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1
    return start


def compress_context(state: AgentState) -> dict[str, Any]:
    """Summarise older messages with one LLM call when the context is too large."""
    messages = list(state.get("messages", []))
    if len(messages) <= config.KEEP_LAST_MESSAGES:
        return {}
    if int(state.get("tool_calls_count", 0)) >= int(
        state.get("max_tool_calls", config.MAX_TOOL_CALLS)
    ):
        return {}

    # Include the system prompt and skill body in the estimate; otherwise a long
    # skill can keep the real request well above 4000 tokens without ever
    # triggering compression.
    system_prompt = build_system_prompt(state)
    if _estimate_tokens([SystemMessage(content=system_prompt)] + messages) <= config.CONTEXT_TOKEN_LIMIT:
        return {}

    # Compress at most once every two iterations; otherwise the summary message
    # itself would be summarised again on every turn.
    iterations = int(state.get("iterations", 0))
    last_compression = max(
        (
            int(step.get("iteration", 0))
            for step in state.get("trajectory", [])
            if step.get("event") == "context_compression"
        ),
        default=-10,
    )
    if iterations - last_compression < 2:
        return {}

    split = _compression_split(messages, config.KEEP_LAST_MESSAGES)
    if split <= 0:
        return {}
    old_messages = messages[:split]
    recent_messages = messages[split:]

    history_lines = []
    for message in old_messages:
        role = message.__class__.__name__.replace("Message", "")
        text = make_snippet(_message_text(message), "", 500)
        if text:
            history_lines.append(f"{role}: {text}")
    history = "\n".join(history_lines)

    summary = ""
    total_tokens = int(state.get("total_tokens", 0))
    try:
        response = get_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Compress a search agent's conversation into a compact progress note. "
                        "Preserve: the user's question, confirmed facts, document IDs that mattered, "
                        "dead ends to avoid, and the current candidate answer. Be concise."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{state.get('question', '')}\n\n"
                        f"Previous summary:\n{state.get('context_summary', '')}\n\n"
                        f"Conversation to compress:\n{history}"
                    )
                ),
            ]
        )
        total_tokens += _usage_tokens(response)
        summary = _content_text(response.content).strip()
        if not summary:
            summary = " | ".join(line[:160] for line in history_lines[-8:])
    except Exception as exc:
        # Deterministic fallback: retain the most useful short fragments.
        summary = (
            f"(Extractive fallback after {type(exc).__name__}) "
            + " | ".join(line[:160] for line in history_lines[-8:])
        )

    summary = _truncate(summary, 1200)
    summary_message = SystemMessage(content=f"[Previous search summary]: {summary}")
    trajectory = list(state.get("trajectory", []))
    trajectory.append(
        {
            "iteration": int(state.get("iterations", 0)),
            "thought": "Compressed older conversation history to control context size.",
            "tool": None,
            "tool_input": None,
            "observation": f"[Previous search summary]: {summary}",
            "event": "context_compression",
        }
    )
    return {
        "messages": [summary_message] + recent_messages,
        "context_summary": summary,
        "total_tokens": total_tokens,
        "trajectory": trajectory,
    }


def compress_context_node(state: AgentState) -> dict[str, Any]:
    return compress_context(state)


# ---------------------------------------------------------------------------
# Finalisation + routing
# ---------------------------------------------------------------------------


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Extract or synthesise the final answer and close the trajectory."""
    messages = list(state.get("messages", []))
    trajectory = list(state.get("trajectory", []))
    total_tokens = int(state.get("total_tokens", 0))
    iterations = int(state.get("iterations", 0))
    answer = ""
    thought = "Extracted final answer from the agent's last message."
    final_latency = 0.0

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            candidate_text = _message_text(message)
            answer = _extract_final_answer(candidate_text)
            if answer:
                thought = candidate_text or thought
                break

    if not answer:
        evidence = _format_evidence_pool(state.get("evidence_pool", []))
        conversation = "\n".join(
            f"{message.__class__.__name__}: {make_snippet(_message_text(message), '', 500)}"
            for message in messages[-8:]
            if _message_text(message)
        )
        budget_note = (
            "The tool-call budget was reached; "
            if state.get("error") == "tool_call_budget_exhausted"
            else ""
        )
        try:
            final_started = time.time()
            response = get_llm().invoke(
                [
                    SystemMessage(
                        content=(
                            "You are finalising a research answer. Use only the evidence and conversation "
                            "provided. Commit to the single most likely answer. "
                            "Reply with exactly one line: FINAL ANSWER: <answer>"
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Question:\n{state.get('question', '')}\n\n"
                            f"{evidence}\n\n{budget_note}Recent conversation:\n{conversation}"
                        )
                    ),
                ]
            )
            final_latency = round(time.time() - final_started, 2)
            total_tokens += _usage_tokens(response)
            content = _content_text(response.content)
            answer = _extract_final_answer(content)
            if not answer:
                if re.search(r"FINAL\s+ANSWER", content, flags=re.IGNORECASE):
                    answer = state.get("candidate_answer", "") or ""
                else:
                    answer = content.strip().splitlines()[0].strip() if content.strip() else ""
                    answer = re.sub(r"^[\"'`*_]+|[\"'`*_]+$", "", answer).strip()
            thought = f"{budget_note}Synthesised the final answer with one LLM call."
        except Exception as exc:
            answer = state.get("candidate_answer", "") or ""
            thought = f"Final synthesis failed ({type(exc).__name__}); used the best candidate."

    if not answer:
        answer = state.get("candidate_answer", "") or "Unknown"

    support_text = " ".join(
        str(entry.get("snippet", "")) for entry in state.get("evidence_pool", [])
    )
    support_text += " " + " ".join(_message_text(message) for message in messages[-8:])
    answer = clean_final_answer(
        str(answer), state.get("question", ""), evidence=support_text
    ) or "Unknown"

    trajectory.append(
        {
            "iteration": iterations,
            "thought": thought,
            "tool": None,
            "tool_input": None,
            "observation": answer,
            "event": "final_answer",
            "latency_seconds": final_latency,
        }
    )
    return {
        "answer": answer,
        "trajectory": trajectory,
        "total_tokens": total_tokens,
        "iterations": iterations,
    }


def route_after_reason(state: AgentState) -> str:
    if state.get("error"):
        return "finalize"
    if state.get("answer"):
        return "finalize"

    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if _has_final_marker(last):
        return "finalize"
    if int(state.get("iterations", 0)) >= int(
        state.get("max_iterations", config.MAX_ITERATIONS)
    ):
        return "finalize"
    if getattr(last, "tool_calls", None):
        return "tools"
    # No tool call and no final marker: nudge the model on the next reason turn.
    return "reason"
