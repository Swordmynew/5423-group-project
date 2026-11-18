"""Run the agent over a question file and write ``test_prediction.jsonl``.

Input (one JSON object per line)::

    {"id": "1005", "question": "..."}

Output (exactly the required fields)::

    {"id": "...", "question": "...", "answer": "...",
     "trajectory": [...], "tool_calls": 4, "total_tokens": 1234}

Examples
--------
python run_test.py --input data/raw/test.jsonl
python run_test.py --input data/raw/test.jsonl --limit 10 --workers 2
python run_test.py --input data/raw/validation.jsonl --output /tmp/dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).absolute().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import config  # noqa: E402


def read_questions(path: Path, limit: int = 0, offset: int = 0) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc

            question_id = record.get("id")
            if question_id is None:
                question_id = record.get("query_id")
            if question_id is None:
                question_id = record.get("question_id")
            question = record.get("question") or record.get("query")
            if question_id is None or not question:
                raise SystemExit(
                    f"Line {line_number} must contain 'id' (or 'query_id'/'question_id') "
                    "and 'question' (or 'query')."
                )
            questions.append({"id": str(question_id), "question": str(question)})

    if offset:
        questions = questions[offset:]
    if limit:
        questions = questions[:limit]
    return questions


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("id") is not None:
                existing[str(record["id"])] = record
    return existing


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list):
        trajectory = []
    return {
        "id": str(record["id"]),
        "question": str(record["question"]),
        "answer": str(record.get("answer", "Unknown")),
        "trajectory": trajectory,
        "tool_calls": int(record.get("tool_calls", 0) or 0),
        "total_tokens": int(record.get("total_tokens", 0) or 0),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL file with id/question fields.")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PREDICTION_PATH,
        help="Destination JSONL (default: ../test_prediction.jsonl).",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent agent runs.")
    parser.add_argument("--resume", action="store_true", help="Skip IDs already present in the output.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    questions = read_questions(input_path, limit=args.limit, offset=args.offset)
    if not questions:
        raise SystemExit("No questions found.")

    existing = read_existing(output_path) if args.resume else {}
    pending = [item for item in questions if item["id"] not in existing]

    print(f"Questions: {len(questions)} | pending: {len(pending)} | workers: {args.workers}")
    print(f"Output: {output_path}")
    if not config.DEEPSEEK_API_KEY:
        print(
            "WARNING: DEEPSEEK_API_KEY is not set. The agent will return 'Unknown'. "
            "Set it in code/.env or the environment before the real run.",
            file=sys.stderr,
        )

    from agent.graph import run_agent

    results: dict[str, dict[str, Any]] = {}
    started = time.time()

    def work(item: dict[str, str]) -> dict[str, Any]:
        try:
            result = run_agent(item["question"], verbose=args.verbose)
        except Exception as exc:  # defensive: one bad question must not stop the batch
            result = {
                "question": item["question"],
                "answer": "Unknown",
                "trajectory": [
                    {
                        "iteration": 0,
                        "thought": f"Unhandled runner error: {type(exc).__name__}: {exc}",
                        "tool": None,
                        "tool_input": None,
                        "observation": str(exc),
                        "event": "error",
                    }
                ],
                "tool_calls": 0,
                "total_tokens": 0,
            }
        record = {
            "id": item["id"],
            "question": item["question"],
            "answer": result.get("answer", "Unknown"),
            "trajectory": result.get("trajectory", []),
            "tool_calls": result.get("tool_calls", 0),
            "total_tokens": result.get("total_tokens", 0),
        }
        return validate_record(record)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(work, item): item for item in pending}
            completed = 0
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = validate_record(
                        {
                            "id": item["id"],
                            "question": item["question"],
                            "answer": "Unknown",
                            "trajectory": [
                                {
                                    "iteration": 0,
                                    "thought": f"Worker failure: {type(exc).__name__}: {exc}",
                                    "tool": None,
                                    "tool_input": None,
                                    "observation": str(exc),
                                    "event": "error",
                                }
                            ],
                            "tool_calls": 0,
                            "total_tokens": 0,
                        }
                    )
                results[record["id"]] = record
                completed += 1
                print(
                    f"[{completed}/{len(pending)}] id={record['id']} "
                    f"tools={record['tool_calls']} tokens={record['total_tokens']} "
                    f"answer={record['answer'][:80]!r}",
                    flush=True,
                )

    ordered: list[dict[str, Any]] = []
    for item in questions:
        if item["id"] in results:
            ordered.append(results[item["id"]])
        elif item["id"] in existing:
            ordered.append(existing[item["id"]])

    write_jsonl(output_path, ordered)
    elapsed = time.time() - started
    total_tools = sum(record["tool_calls"] for record in ordered)
    total_tokens = sum(record["total_tokens"] for record in ordered)
    print(
        f"\nWrote {len(ordered)} records to {output_path} in {elapsed:.1f}s "
        f"({total_tools} tool calls, {total_tokens:,} tokens)."
    )
    if len(ordered) != len(questions):
        print("WARNING: output record count does not match the input question count.", file=sys.stderr)


if __name__ == "__main__":
    main()
