"""Literal / full-text search tool for finding exact facts inside documents."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from langchain_core.tools import tool

import config
from index.retriever import get_retriever
from index.text_utils import collapse_whitespace, make_snippet, tokenize_text
from tools.cache import get_document_text

_WORD_BOUNDARY = re.compile(r"[A-Za-z0-9]+")


def _context_snippet(text: str, start: int, end: int, context_chars: int) -> str:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    snippet = collapse_whitespace(text[left:right])
    if left > 0:
        snippet = "…" + snippet
    if right < len(text):
        snippet = snippet + "…"
    return snippet[: config.MAX_SNIPPET_CHARS]


def _search_in_document(text: str, query: str, max_results: int) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for match in pattern.finditer(text):
        if len(matches) >= max_results:
            break
        matches.append(
            {
                "offset": match.start(),
                "snippet": _context_snippet(text, match.start(), match.end(), config.TEXT_SEARCH_CONTEXT_CHARS),
            }
        )
    if matches:
        return matches

    # Fall back to the distinctive query terms when the exact phrase is split
    # across a line break or punctuation.
    terms = [term for term in tokenize_text(query) if len(term) > 2][:6]
    if not terms:
        return []
    term_pattern = re.compile(r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b", re.IGNORECASE)
    for match in term_pattern.finditer(text):
        if len(matches) >= max_results:
            break
        matches.append(
            {
                "offset": match.start(),
                "snippet": _context_snippet(text, match.start(), match.end(), config.TEXT_SEARCH_CONTEXT_CHARS),
            }
        )
    return matches


@tool
def text_searcher(query: str, doc_id: str = "", max_results: int = config.TEXT_SEARCH_MAX_RESULTS) -> str:
    """Find an exact phrase inside one document or across the corpus.

    Use it to verify a name/date, search inside a long document, or try an
    unusual quoted phrase when ranked retrieval is weak.

    Args:
        query: Exact string or a few distinctive words.
        doc_id: Optional document ID; empty means corpus-wide FTS.
        max_results: Matches to return, 1-10. Default 4.

    Returns:
        JSON with ``matches``: doc_id, title, offset, and <=160-char snippet.
    """
    query = (query or "").strip()
    if not query:
        return json.dumps({"error": "query must be non-empty", "matches": []}, ensure_ascii=False)
    max_results = max(1, min(int(max_results or 5), 10))

    try:
        retriever = get_retriever()
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "matches": []}, ensure_ascii=False)

    if not retriever.db_path.exists():
        return json.dumps(
            {"error": f"Corpus database not found: {retriever.db_path}", "matches": []},
            ensure_ascii=False,
        )

    # Single-document search is exact, fast, and needs no FTS index.
    if doc_id:
        document = retriever.get_document(doc_id)
        if not document:
            return json.dumps({"error": f"Document '{doc_id}' was not found.", "matches": []}, ensure_ascii=False)
        matches = _search_in_document(
            get_document_text(doc_id) or document.get("text") or "",
            query,
            max_results,
        )
        for match in matches:
            match["doc_id"] = document["docid"]
            match["title"] = document.get("title", "")
        return json.dumps(
            {"query": query, "doc_id": document["docid"], "count": len(matches), "matches": matches},
            ensure_ascii=False,
        )

    connection = sqlite3.connect(str(retriever.db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        has_fts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='docs_fts'"
        ).fetchone() is not None

        if has_fts:
            terms = [term for term in _WORD_BOUNDARY.findall(query.lower()) if len(term) > 1][:8]
            if not terms:
                return json.dumps({"query": query, "count": 0, "matches": []}, ensure_ascii=False)
            fts_query = " OR ".join(f'"{term}"' for term in terms)
            rows = connection.execute(
                "SELECT docs_fts.docid AS docid, "
                "snippet(docs_fts, 2, '[', ']', '…', 12) AS snip, "
                "docs.title AS title, docs.url AS url "
                "FROM docs_fts JOIN docs ON docs.docid = docs_fts.docid "
                "WHERE docs_fts MATCH ? LIMIT ?",
                (fts_query, max_results),
            ).fetchall()
            matches = [
                {
                    "doc_id": row["docid"],
                    "title": row["title"] or "",
                    "url": row["url"] or "",
                    "snippet": make_snippet(row["snip"] or "", query, config.MAX_SNIPPET_CHARS),
                }
                for row in rows
            ]
        else:
            pattern = f"%{query}%"
            rows = connection.execute(
                "SELECT docid, title, url, text FROM docs WHERE text LIKE ? LIMIT ?",
                (pattern, max_results),
            ).fetchall()
            matches = [
                {
                    "doc_id": row["docid"],
                    "title": row["title"] or "",
                    "url": row["url"] or "",
                    "snippet": make_snippet(row["text"], query, config.MAX_SNIPPET_CHARS),
                }
                for row in rows
            ]
    except sqlite3.Error as exc:
        return json.dumps({"error": f"Search failed: {exc}", "matches": []}, ensure_ascii=False)
    finally:
        connection.close()

    return json.dumps(
        {"query": query, "count": len(matches), "matches": matches},
        ensure_ascii=False,
    )
