"""Bilingual Gradio UI for the skill-augmented search agent.

Features
--------
* English / 中文 label switching
* Fast (5 tools / 6 iterations) and Balanced (8 tools / 10 iterations) modes
* Answer, statistics, selected skills, evidence, trajectory, step timeline, raw JSON
* Per-step LLM/tool latency so slow runs are diagnosable
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).absolute().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import config  # noqa: E402

EVIDENCE_HEADERS = ["doc_id", "score", "snippet", "url", "sources"]
TRAJECTORY_HEADERS = [
    "iteration", "thought", "tool", "tool_input", "observation",
    "llm_s", "tool_s",
]

MODES = {
    "balanced": {"max_iterations": 10, "max_tool_calls": 8},
    "fast": {"max_iterations": 6, "max_tool_calls": 5},
}

T = {
    "en": {
        "title": "# COMP5423 Skill-Augmented Search Agent",
        "subtitle": (
            "Complex questions over BrowseComp-Plus with BM25S + BGE-M3 + "
            "Qwen3 retrieval, RRF, four tools, reusable skills, and a LangGraph loop."
        ),
        "status": "**Status**",
        "question_label": "Question",
        "question_placeholder": "Ask a multi-clue research question ...",
        "mode_label": "Mode",
        "balanced": "Balanced · 8 tools / 10 iterations · better accuracy",
        "fast": "Fast · 5 tools / 6 iterations · lower latency",
        "run": "Search & Answer",
        "clear": "Clear",
        "answer": "### Answer",
        "answer_empty": "_The answer will appear here._",
        "stats_empty": "_Tool calls, tokens, iterations and latency will appear here._",
        "skills_empty": "_Selected skills will appear here._",
        "evidence_tab": "Evidence / 证据",
        "trajectory_tab": "Trajectory / 轨迹",
        "timeline_tab": "Timeline / 时间线",
        "raw_tab": "Raw JSON / 原始记录",
        "evidence_label": "Evidence pool (top documents)",
        "trajectory_label": "Chronological agent trajectory",
        "timeline_label": "",
        "json_label": "Complete record",
        "timeline_empty": "<p style='color:#777'>No steps yet.</p>",
        "no_evidence": "No evidence collected.",
        "tools": "Tools",
        "tokens": "Tokens",
        "iterations": "Iterations",
        "time": "Time",
        "mode": "Mode",
        "skills": "Skill(s)",
        "none": "none",
        "empty_question": "Please enter a question.",
        "error": "Error",
    },
    "zh": {
        "title": "# COMP5423 技能增强搜索 Agent",
        "subtitle": (
            "在 BrowseComp-Plus 上结合 BM25S、BGE-M3、Qwen3 检索、RRF、"
            "四个工具、可复用 Skill 和 LangGraph 迭代循环回答复杂问题。"
        ),
        "status": "**运行状态**",
        "question_label": "问题",
        "question_placeholder": "输入一个多线索研究问题 ...",
        "mode_label": "模式",
        "balanced": "均衡 · 8 次工具 / 10 轮 · 精度更高",
        "fast": "快速 · 5 次工具 / 6 轮 · 延迟更低",
        "run": "搜索并回答",
        "clear": "清空",
        "answer": "### 答案",
        "answer_empty": "_答案会显示在这里。_",
        "stats_empty": "_工具调用、tokens、迭代次数和耗时会显示在这里。_",
        "skills_empty": "_选中的 Skill 会显示在这里。_",
        "evidence_tab": "Evidence / 证据",
        "trajectory_tab": "Trajectory / 轨迹",
        "timeline_tab": "Timeline / 时间线",
        "raw_tab": "Raw JSON / 原始记录",
        "evidence_label": "证据池（最相关文档）",
        "trajectory_label": "Agent 完整轨迹",
        "timeline_label": "",
        "json_label": "完整记录",
        "timeline_empty": "<p style='color:#777'>还没有执行步骤。</p>",
        "no_evidence": "没有收集到证据。",
        "tools": "工具调用",
        "tokens": "Tokens",
        "iterations": "迭代",
        "time": "耗时",
        "mode": "模式",
        "skills": "Skill",
        "none": "无",
        "empty_question": "请输入问题。",
        "error": "错误",
    },
}


def _short(value: Any, limit: int = 320) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _status_md(lang: str) -> str:
    labels = T[lang]
    try:
        from index.retriever import get_retriever

        stats = get_retriever().stats()
        docs = stats.get("documents", "?")
        chunks = stats.get("chunks", "?")
        docs_text = f"{docs:,}" if isinstance(docs, int) else str(docs)
        chunks_text = f"{chunks:,}" if isinstance(chunks, int) else str(chunks)
        methods = ", ".join(stats.get("available_methods", [])) or "none"
        model = config.DEEPSEEK_MODEL
        return (
            f"{labels['status']}  \n"
            f"**Corpus:** {docs_text} docs · {chunks_text} chunks  \n"
            f"**Indexes:** {methods}  \n"
            f"**LLM:** `{model}`  \n"
            f"**Vector windows:** `{config.DENSE_DOC_WINDOWS}`"
        )
    except Exception as exc:
        return f"{labels['status']}  \n`{type(exc).__name__}: {exc}`"


def _trajectory_rows(trajectory: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for step in trajectory or []:
        rows.append(
            [
                step.get("iteration", ""),
                _short(step.get("thought", ""), 500),
                step.get("tool") or "",
                _short(step.get("tool_input", ""), 300),
                _short(step.get("observation", ""), 500),
                step.get("latency_seconds", ""),
                step.get("tool_seconds", ""),
            ]
        )
    return rows


def _evidence_rows(pool: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for entry in pool or []:
        rows.append(
            [
                entry.get("doc_id", ""),
                round(float(entry.get("relevance_score", 0.0)), 5),
                _short(entry.get("snippet", ""), 300),
                entry.get("url", ""),
                ", ".join(entry.get("sources", [])),
            ]
        )
    return rows


def _timeline_html(trajectory: list[dict[str, Any]], lang: str) -> str:
    if not trajectory:
        return T[lang]["timeline_empty"]

    blocks = []
    for index, step in enumerate(trajectory, start=1):
        tool = step.get("tool") or "reason"
        event = step.get("event") or ""
        latency = step.get("latency_seconds")
        tool_seconds = step.get("tool_seconds")
        timing = []
        if latency not in (None, "", 0, 0.0):
            timing.append(f"LLM {float(latency):.1f}s")
        if tool_seconds not in (None, "", 0, 0.0):
            timing.append(f"tool {float(tool_seconds):.2f}s")
        timing_text = " · ".join(timing) or "instant"
        thought = html.escape(_short(step.get("thought", ""), 700))
        observation = html.escape(_short(step.get("observation", ""), 700))
        tool_input = html.escape(_short(step.get("tool_input", ""), 400))
        event_badge = f" <span style='color:#888'>[{html.escape(event)}]</span>" if event else ""
        blocks.append(
            "<div style='border:1px solid #e5e7eb;border-radius:10px;padding:10px 14px;"
            "margin:8px 0;background:#fafafa'>"
            f"<div style='font-weight:600;color:#1f2937'>Step {index} · iteration "
            f"{step.get('iteration', '?')} · {html.escape(str(tool))}{event_badge}"
            f" <span style='float:right;color:#6b7280;font-weight:400'>{timing_text}</span></div>"
            f"<div style='margin-top:6px'><b>Thought</b><br>{thought or '<i>—</i>'}</div>"
            f"<div style='margin-top:6px'><b>Tool input</b><br><code>{tool_input or '—'}</code></div>"
            f"<div style='margin-top:6px'><b>Observation</b><br>{observation or '<i>—</i>'}</div>"
            "</div>"
        )
    return "<div>" + "".join(blocks) + "</div>"


def run_question(question: str, mode: str, lang: str):
    labels = T.get(lang, T["en"])
    question = (question or "").strip()
    if not question:
        return (
            f"{labels['answer']}\n\n**{labels['empty_question']}**",
            labels["stats_empty"],
            labels["skills_empty"],
            [],
            [],
            labels["timeline_empty"],
            {"error": labels["empty_question"]},
        )

    budget = MODES.get(mode, MODES["balanced"])
    try:
        from agent.graph import run_agent

        result = run_agent(
            question,
            verbose=True,
            max_iterations=budget["max_iterations"],
            max_tool_calls=budget["max_tool_calls"],
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return (
            f"{labels['answer']}\n\n**{labels['error']}:** {message}",
            labels["stats_empty"],
            labels["skills_empty"],
            [],
            [],
            labels["timeline_empty"],
            {"error": message},
        )

    answer = result.get("answer", "Unknown")
    answer_md = f"{labels['answer']}\n\n**{answer}**"
    stats_md = (
        f"**{labels['tools']}:** {result.get('tool_calls', 0)} &nbsp;|&nbsp; "
        f"**{labels['tokens']}:** {result.get('total_tokens', 0):,} &nbsp;|&nbsp; "
        f"**{labels['iterations']}:** {result.get('iterations', 0)} &nbsp;|&nbsp; "
        f"**{labels['time']}:** {result.get('elapsed_seconds', 0)}s &nbsp;|&nbsp; "
        f"**{labels['mode']}:** {labels.get(mode, mode)}"
    )
    skills = ", ".join(result.get("active_skills", [])) or labels["none"]
    skills_md = f"**{labels['skills']}:** {skills}"
    trajectory = result.get("trajectory", [])
    return (
        answer_md,
        stats_md,
        skills_md,
        _evidence_rows(result.get("evidence_pool", [])),
        _trajectory_rows(trajectory),
        _timeline_html(trajectory, lang),
        result,
    )


def clear_all(lang: str):
    labels = T.get(lang, T["en"])
    return (
        labels["answer_empty"],
        labels["stats_empty"],
        labels["skills_empty"],
        [],
        [],
        labels["timeline_empty"],
        {},
    )


def switch_language(lang: str, mode: str):
    labels = T.get(lang, T["en"])
    mode = mode if mode in MODES else "balanced"
    return (
        gr.update(value=labels["title"]),
        gr.update(value=labels["subtitle"]),
        gr.update(value=_status_md(lang)),
        gr.update(label=labels["question_label"], placeholder=labels["question_placeholder"]),
        gr.update(
            label=labels["mode_label"],
            choices=[(labels["balanced"], "balanced"), (labels["fast"], "fast")],
            value=mode,
        ),
        gr.update(value=labels["run"]),
        gr.update(value=labels["clear"]),
        gr.update(value=labels["answer_empty"]),
        gr.update(value=labels["stats_empty"]),
        gr.update(value=labels["skills_empty"]),
        gr.update(label=labels["evidence_label"]),
        gr.update(label=labels["trajectory_label"]),
        gr.update(value=labels["timeline_empty"]),
        gr.update(label=labels["json_label"]),
    )


def build_demo():
    global gr
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Gradio is required. `pip install -r requirements.txt`.") from exc

    lang0 = "en"
    labels = T[lang0]

    with gr.Blocks(title="COMP5423 Skill-Augmented Search Agent") as demo:
        with gr.Row():
            with gr.Column(scale=2):
                lang_radio = gr.Radio(
                    choices=[("English", "en"), ("中文", "zh")],
                    value=lang0,
                    label="Language / 语言",
                )
            with gr.Column(scale=3):
                mode_radio = gr.Radio(
                    choices=[(labels["balanced"], "balanced"), (labels["fast"], "fast")],
                    value="balanced",
                    label=labels["mode_label"],
                )

        header = gr.Markdown(labels["title"])
        subtitle = gr.Markdown(labels["subtitle"])
        status = gr.Markdown(_status_md(lang0))
        question_box = gr.Textbox(
            label=labels["question_label"],
            placeholder=labels["question_placeholder"],
            lines=4,
        )
        with gr.Row():
            with gr.Column(scale=3):
                run_button = gr.Button(labels["run"], variant="primary")
            with gr.Column(scale=1):
                clear_button = gr.Button(labels["clear"])

        answer_box = gr.Markdown(labels["answer_empty"])
        with gr.Row():
            with gr.Column(scale=2):
                stats_box = gr.Markdown(labels["stats_empty"])
            with gr.Column(scale=1):
                skills_box = gr.Markdown(labels["skills_empty"])

        with gr.Tab(labels["evidence_tab"]):
            evidence_table = gr.Dataframe(
                headers=EVIDENCE_HEADERS,
                label=labels["evidence_label"],
                wrap=True,
            )
        with gr.Tab(labels["trajectory_tab"]):
            trajectory_table = gr.Dataframe(
                headers=TRAJECTORY_HEADERS,
                label=labels["trajectory_label"],
                wrap=True,
            )
        with gr.Tab(labels["timeline_tab"]):
            timeline_html = gr.HTML(labels["timeline_empty"])
        with gr.Accordion(labels["raw_tab"], open=False):
            raw_json = gr.JSON(label=labels["json_label"])

        gr.Examples(
            examples=[
                ["Which researcher, born in the 1910s, discovered a compound used to treat a metal-storage disease and also competed in a national chess tournament?"],
                ["An institute founded in the 1950s in a capital city is named after a scientist who refused a patent for a medical discovery. Which institute is it?"],
            ],
            inputs=question_box,
        )

        outputs = [
            answer_box, stats_box, skills_box,
            evidence_table, trajectory_table, timeline_html, raw_json,
        ]
        run_button.click(
            fn=run_question,
            inputs=[question_box, mode_radio, lang_radio],
            outputs=outputs,
        )
        question_box.submit(
            fn=run_question,
            inputs=[question_box, mode_radio, lang_radio],
            outputs=outputs,
        )
        clear_button.click(
            fn=clear_all,
            inputs=[lang_radio],
            outputs=outputs,
        )
        lang_radio.change(
            fn=switch_language,
            inputs=[lang_radio, mode_radio],
            outputs=[
                header, subtitle, status, question_box, mode_radio,
                run_button, clear_button, answer_box, stats_box, skills_box,
                evidence_table, trajectory_table, timeline_html, raw_json,
            ],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default=config.GRADIO_SERVER_NAME)
    parser.add_argument("--server-port", type=int, default=config.GRADIO_SERVER_PORT)
    parser.add_argument("--share", action="store_true", default=config.GRADIO_SHARE)
    args = parser.parse_args()

    demo = build_demo()
    demo.queue()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
