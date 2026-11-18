"""Metadata filtering tool for narrowing the corpus by source/date/length."""

from __future__ import annotations

import json
import sqlite3

from langchain_core.tools import tool

from index.retriever import get_retriever


@tool
def metadata_filter(
    domain_contains: str = "",
    title_contains: str = "",
    url_contains: str = "",
    date_from: str = "",
    date_to: str = "",
    min_words: int = 0,
    max_words: int = 0,
    limit: int = 10,
) -> str:
    """Filter the corpus by domain, title, URL, date, or document length.

    Use it when a clue names a source, website, publication year, or document
    type; open the returned doc_ids with document_reader/text_searcher.

    Args:
        domain_contains/title_contains/url_contains: substring filters.
        date_from/date_to: ISO dates "YYYY-MM-DD".
        min_words/max_words: document length bounds (0 = no bound).
        limit: maximum documents, 1-20. Default 10.

    Returns:
        JSON with ``count`` and metadata rows: doc_id, title, url, domain,
        date, n_words. No full text is returned.
    """
    clauses: list[str] = []
    parameters: list[object] = []

    if domain_contains:
        clauses.append("domain LIKE ?")
        parameters.append(f"%{domain_contains.strip().lower()}%")
    if title_contains:
        clauses.append("title LIKE ?")
        parameters.append(f"%{title_contains.strip()}%")
    if url_contains:
        clauses.append("url LIKE ?")
        parameters.append(f"%{url_contains.strip()}%")
    if date_from:
        clauses.append("date >= ?")
        parameters.append(date_from.strip())
    if date_to:
        clauses.append("date <= ?")
        parameters.append(date_to.strip())
    if min_words and min_words > 0:
        clauses.append("n_words >= ?")
        parameters.append(int(min_words))
    if max_words and max_words > 0:
        clauses.append("n_words <= ?")
        parameters.append(int(max_words))

    if not clauses:
        return json.dumps(
            {
                "error": "Provide at least one filter (domain_contains, title_contains, url_contains, "
                "date_from, date_to, min_words, or max_words).",
                "results": [],
            },
            ensure_ascii=False,
        )

    limit = max(1, min(int(limit or 10), 20))
    query = (
        "SELECT docid, title, url, domain, date, n_words FROM docs WHERE "
        + " AND ".join(clauses)
        + " ORDER BY n_words DESC LIMIT ?"
    )
    parameters.append(limit)

    try:
        retriever = get_retriever()
        if not retriever.db_path.exists():
            return json.dumps({"error": "Corpus database not found.", "results": []}, ensure_ascii=False)
        connection = sqlite3.connect(str(retriever.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(query, parameters).fetchall()
        finally:
            connection.close()
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "results": []}, ensure_ascii=False)

    results = [
        {
            "doc_id": row["docid"],
            "title": row["title"] or "",
            "url": row["url"] or "",
            "domain": row["domain"] or "",
            "date": row["date"] or "",
            "n_words": row["n_words"],
        }
        for row in rows
    ]
    return json.dumps(
        {
            "filters": {
                "domain_contains": domain_contains,
                "title_contains": title_contains,
                "url_contains": url_contains,
                "date_from": date_from,
                "date_to": date_to,
                "min_words": min_words,
                "max_words": max_words,
            },
            "count": len(results),
            "results": results,
        },
        ensure_ascii=False,
    )
