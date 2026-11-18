"""Evaluate the agent on a labelled JSONL set with normalised exact match (EM).

The official metric is used exactly as specified: lowercase, remove English
articles, strip punctuation, collapse whitespace, then compare strings.

Examples
--------
python evaluate.py --input data/raw/validation.jsonl
python evaluate.py --input data/raw/validation.jsonl --limit 20 --concurrency 4
python evaluate.py --input data/raw/validation.jsonl --predictions ../test_prediction.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import string
import sys
import time
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).absolute().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import config  # noqa: E402


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def normalized_em(prediction: str, gold: str) -> bool:
    return normalize(prediction) == normalize(gold)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def select_split(records: list[dict[str, Any]], offset: int, limit: int) -> list[dict[str, Any]]:
    if offset:
        records = records[offset:]
    if limit:
        records = records[:limit]
    return records


async def run_one(
    item: dict[str, Any], semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    from agent.graph import run_agent

    async with semaphore:
        result = await asyncio.to_thread(run_agent, str(item.get("question", "")), False)
    result["_id"] = str(item.get("id"))
    return result


async def evaluate_with_agent(
    questions: list[dict[str, Any]], concurrency: int
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        asyncio.create_task(run_one(item, semaphore))
        for item in questions
    ]
    results: list[dict[str, Any]] = []
    completed = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        completed += 1
        print(
            f"[{completed}/{len(tasks)}] {result.get('answer', '')[:80]!r} "
            f"tools={result.get('tool_calls', 0)} tokens={result.get('total_tokens', 0)}",
            flush=True,
        )
    return results


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if record.get("id") is not None:
            predictions[str(record["id"])] = record
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Labelled JSONL with id/question/answer.")
    parser.add_argument("--predictions", type=Path, default=None, help="Evaluate an existing prediction file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PROCESSED_DIR / "eval_predictions.jsonl",
        help="Where to save predictions when running the agent.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    gold_records = select_split(read_jsonl(input_path), args.offset, args.limit)
    if not gold_records:
        raise SystemExit("No evaluation records found.")

    started = time.time()
    predictions: dict[str, dict[str, Any]] = {}
    if args.predictions:
        predictions = load_predictions(Path(args.predictions).expanduser().resolve())
        print(f"Loaded {len(predictions)} existing predictions.")
    else:
        if not config.DEEPSEEK_API_KEY:
            print(
                "WARNING: DEEPSEEK_API_KEY is not set; every answer will be 'Unknown'.",
                file=sys.stderr,
            )
        results = asyncio.run(evaluate_with_agent(gold_records, args.concurrency))
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for result in results:
                record = {
                    "id": str(result.get("_id")),
                    "question": str(result.get("question", "")),
                    "answer": str(result.get("answer", "Unknown")),
                    "trajectory": result.get("trajectory", []),
                    "tool_calls": int(result.get("tool_calls", 0)),
                    "total_tokens": int(result.get("total_tokens", 0)),
                }
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                predictions[record["id"]] = record
        print(f"Saved predictions to {output_path}")

    rows = []
    correct = 0
    for gold in gold_records:
        question_id = str(gold.get("id"))
        prediction = predictions.get(question_id, {})
        predicted_answer = str(prediction.get("answer", ""))
        gold_answer = str(gold.get("answer", ""))
        is_correct = normalized_em(predicted_answer, gold_answer)
        correct += int(is_correct)
        rows.append(
            {
                "id": question_id,
                "gold": gold_answer,
                "prediction": predicted_answer,
                "em": is_correct,
                "tool_calls": int(prediction.get("tool_calls", 0)),
                "total_tokens": int(prediction.get("total_tokens", 0)),
            }
        )

    total = len(rows)
    em = correct / total if total else 0.0
    avg_tools = sum(row["tool_calls"] for row in rows) / total if total else 0.0
    avg_tokens = sum(row["total_tokens"] for row in rows) / total if total else 0.0
    elapsed = time.time() - started

    print("\n" + "=" * 72)
    print(f"Examples:        {total}")
    print(f"Normalised EM:   {em:.4f}  ({correct}/{total})")
    print(f"Avg tool calls:  {avg_tools:.2f}")
    print(f"Avg tokens:      {avg_tokens:,.1f}")
    print(f"Total tokens:    {sum(row['total_tokens'] for row in rows):,}")
    print(f"Elapsed:         {elapsed:.1f}s")
    print("=" * 72)

    if args.verbose:
        for row in rows:
            status = "OK " if row["em"] else "ERR"
            print(f"{status} id={row['id']} gold={row['gold']!r} pred={row['prediction']!r}")

    detail_path = Path(args.output).with_name("eval_details.jsonl")
    try:
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with detail_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Per-question details: {detail_path}")
    except OSError as exc:
        print(f"Could not write details file: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
