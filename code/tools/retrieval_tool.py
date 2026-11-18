"""The single retrieval tool exposed to the agent.

BM25, BGE-M3 and Qwen3-Embedding are fused *inside* the retriever; the agent
never receives separate tools for them.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

import config
from index.retriever import get_retriever


@tool
def unified_retrieval(query: str, k: int = config.DEFAULT_RETRIEVAL_K, method: str = "hybrid") -> str:
    """Search the fixed BrowseComp-Plus corpus and return ranked documents.

    Use this for every new clue or sub-query; search 2-8 distinctive words, not
    the whole question. All retrieval backends are fused internally by RRF.

    Args:
        query: Focused keywords, name, date, or short phrase.
        k: Number of documents, 1-20. Default 4.
        method: "hybrid" (default), "bm25", "bge", or "qwen3".

    Returns:
        JSON with ``results``: doc_id, score, <=160-char snippet, url, title,
        date, and sources. Failures return ``error`` and an empty list.
    """
    query = (query or "").strip()
    if not query:
        return json.dumps({"error": "query must be non-empty", "results": []}, ensure_ascii=False)

    try:
        retriever = get_retriever()
        results = retriever.search(query=query, k=k, method=method)
        payload = {
            "query": query,
            "method_requested": method,
            "results": results,
        }
    except Exception as exc:  # pragma: no cover - depends on local indexes
        payload = {
            "query": query,
            "error": f"{type(exc).__name__}: {exc}",
            "results": [],
        }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
