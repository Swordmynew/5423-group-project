#!/usr/bin/env python3
"""Heuristic helper for the multi-clue-decomposer skill.

This is an offline aid, not a required part of the agent loop.  It prints a
clue table skeleton and suggested search queries for a question.

Usage:
    python scripts/decompose.py "A scientist who ... and ..."
    echo "A scientist who ..." | python scripts/decompose.py
"""

from __future__ import annotations

import json
import re
import sys

SPLIT_RE = re.compile(r"(?:;\s*|\.\s+(?=[A-Z(])|\n+|^\s*[-*\u2022]\s*)", re.MULTILINE)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "who", "whom", "whose", "which",
    "that", "was", "were", "is", "are", "be", "been", "being", "in", "on",
    "at", "by", "for", "from", "with", "as", "of", "to", "their", "they",
    "them", "he", "she", "his", "her", "it", "its", "this", "these", "those",
    "also", "later", "same", "during", "after", "before", "when", "where",
    "what", "asked", "known", "called", "had", "has", "have", "did", "does",
}


def split_clues(question: str) -> list[str]:
    text = question.strip()
    if not text:
        return []
    pieces = [piece.strip(" -\t") for piece in SPLIT_RE.split(text)]
    clues = []
    for piece in pieces:
        if len(piece.split()) < 3:
            continue
        if not piece.endswith(("?", ".")):
            piece = piece.rstrip(", ")
        clues.append(piece)
    if not clues:
        clues = [text]
    return clues


def suggest_query(clue: str, max_terms: int = 7) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", clue)
    terms = []
    for word in words:
        lowered = word.lower().strip("'")
        if lowered in STOPWORDS or len(lowered) < 2:
            continue
        if word not in terms:
            terms.append(word)
    return " ".join(terms[:max_terms])


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = sys.stdin.read()

    clues = split_clues(question)
    rows = [
        {
            "index": index,
            "clue": clue,
            "suggested_query": suggest_query(clue),
            "status": "open",
        }
        for index, clue in enumerate(clues, start=1)
    ]
    print(
        json.dumps(
            {
                "question": question.strip(),
                "target_type": "fill in: person / organisation / place / date / number / title",
                "clues": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
