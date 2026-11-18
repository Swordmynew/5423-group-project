"""Small in-process cache for text extracted from PDFs.

``document_reader`` fills this cache; ``text_searcher`` consults it so that a
PDF opened once can be searched with short snippets without keeping the full
text in the chat history.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

_MAX_ENTRIES = 16
_LOCK = threading.Lock()
_CACHE: "OrderedDict[str, str]" = OrderedDict()


def set_document_text(doc_id: str, text: str) -> None:
    if not doc_id or not text:
        return
    with _LOCK:
        _CACHE[doc_id] = text
        _CACHE.move_to_end(doc_id)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)


def get_document_text(doc_id: str) -> str:
    if not doc_id:
        return ""
    with _LOCK:
        text = _CACHE.get(doc_id)
        if text:
            _CACHE.move_to_end(doc_id)
        return text or ""
