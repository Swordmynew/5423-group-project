"""Document opening tool with an optional PDF fallback (used by the PDF skill)."""

from __future__ import annotations

import io
import json
import threading
import urllib.request
from typing import Any

from langchain_core.tools import tool

import config
from index.retriever import get_retriever
from index.text_utils import focused_excerpt
from tools.cache import set_document_text

_PDF_CACHE: dict[str, str] = {}
_PDF_CACHE_ORDER: list[str] = []
_PDF_LOCK = threading.Lock()
_PDF_CACHE_LIMIT = 16
_PDF_MAX_BYTES = 30 * 1024 * 1024


def _extract_pdf_bytes(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pypdf is required for PDF extraction. Install it with `pip install pypdf`."
        ) from exc

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def _fetch_pdf_text(url: str) -> str:
    with _PDF_LOCK:
        if url in _PDF_CACHE:
            return _PDF_CACHE[url]

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; COMP5423SearchAgent/1.0; "
                "+https://www.polyu.edu.hk/)"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(_PDF_MAX_BYTES)

    if not data.startswith(b"%PDF"):
        raise ValueError("The URL did not return a PDF file.")

    text = _extract_pdf_bytes(data)
    with _PDF_LOCK:
        _PDF_CACHE[url] = text
        _PDF_CACHE_ORDER.append(url)
        while len(_PDF_CACHE_ORDER) > _PDF_CACHE_LIMIT:
            oldest = _PDF_CACHE_ORDER.pop(0)
            _PDF_CACHE.pop(oldest, None)
    return text


@tool
def document_reader(
    doc_id: str,
    max_chars: int = config.DOC_READER_DEFAULT_CHARS,
    focus: str = "",
    extract_pdf: bool = False,
) -> str:
    """Open one corpus document and return a bounded, focused excerpt.

    Use after retrieval finds a promising document, or with ``extract_pdf`` for
    a PDF source. The full document is never placed in the message list.

    Args:
        doc_id: Document ID from a search/filter tool.
        max_chars: Excerpt length, 200-4000. Default 1000.
        focus: Optional words used to centre the excerpt.
        extract_pdf: Download and parse the PDF when needed.

    Returns:
        JSON with doc_id, title, url, date, total_chars, excerpt, truncated.
    """
    doc_id = str(doc_id or "").strip()
    if not doc_id:
        return json.dumps({"error": "doc_id must be non-empty"}, ensure_ascii=False)

    max_chars = max(200, min(int(max_chars or config.DOC_READER_DEFAULT_CHARS), config.DOC_READER_MAX_CHARS))
    try:
        document = get_retriever().get_document(doc_id)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
    if not document:
        return json.dumps({"error": f"Document '{doc_id}' was not found."}, ensure_ascii=False)

    text = document.get("text") or ""
    url = document.get("url") or ""
    pdf_note = ""
    if extract_pdf and (url.lower().endswith(".pdf") or len(text.strip()) < 80):
        try:
            text = _fetch_pdf_text(url)
            pdf_note = "Text extracted from PDF."
            set_document_text(doc_id, text)
        except Exception as exc:
            pdf_note = f"PDF extraction failed: {type(exc).__name__}: {exc}"

    excerpt: str = focused_excerpt(text, focus=focus, max_chars=max_chars)
    payload: dict[str, Any] = {
        "doc_id": document["docid"],
        "title": document.get("title", ""),
        "url": url,
        "domain": document.get("domain", ""),
        "date": document.get("date", ""),
        "total_chars": len(text),
        "excerpt": excerpt,
        "truncated": len(text) > len(excerpt),
    }
    if pdf_note:
        payload["note"] = pdf_note
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
