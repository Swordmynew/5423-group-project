"""Shared text helpers: BM25 tokenisation and short evidence snippets."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")

# A compact English stopword list; used only for sparse retrieval.  Keeping it
# local avoids a hard dependency on an NLP toolkit.
_FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}

try:  # bm25s ships a larger list; use it when available
    from bm25s.stopwords import STOPWORDS_EN as _BM25_STOPWORDS

    STOPWORDS = set(_BM25_STOPWORDS)
except Exception:  # pragma: no cover - bm25s is optional at import time
    STOPWORDS = _FALLBACK_STOPWORDS


def tokenize_text(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation with stopword/one-char removal.

    The same function is used when building and querying the BM25 index, which
    is essential for consistent term matching.
    """
    if not text:
        return []
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def make_snippet(text: str, query: str = "", max_chars: int = 200) -> str:
    """Return a <= ``max_chars`` character snippet centred on query terms."""
    if not text:
        return ""
    flat = collapse_whitespace(text)
    if len(flat) <= max_chars:
        return flat

    query_tokens = [token for token in TOKEN_RE.findall((query or "").lower()) if len(token) > 2]
    centre = 0
    lowered = flat.lower()
    for token in query_tokens:
        position = lowered.find(token)
        if position >= 0:
            centre = position
            break

    start = max(0, centre - max_chars // 3)
    end = min(len(flat), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(flat) else ""
    available = max_chars - len(prefix) - len(suffix)
    body = flat[start:end][:available]
    return (prefix + body + suffix)[:max_chars]


def focused_excerpt(
    text: str,
    focus: str,
    max_chars: int,
    *,
    default_start: int = 0,
) -> str:
    """Return a bounded excerpt around the first useful ``focus`` token."""
    if not text:
        return ""
    max_chars = max(200, max_chars)
    if len(text) <= max_chars:
        return text

    focus_tokens = [t for t in TOKEN_RE.findall((focus or "").lower()) if len(t) > 2]
    lowered = text.lower()
    position = -1
    for token in focus_tokens:
        candidate = lowered.find(token)
        if candidate >= 0:
            position = candidate
            break
    if position < 0:
        position = default_start

    start = max(0, position - max_chars // 3)
    end = min(len(text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    return text[start:end]


def clean_final_answer(answer: str, question: str, evidence: str = "") -> str:
    """Remove common answer-format noise without changing the entity.

    The official metric compares strings exactly (after punctuation/article
    normalization), so parenthetical disambiguation, middle names when the
    question explicitly asks for first and last name, and title subtitles are
    costly.  This helper only removes those formatting additions.
    """
    if not answer:
        return ""
    cleaned = collapse_whitespace(answer).strip().strip("\"'` ").strip()
    # Drop one or more trailing parenthetical/bracketed qualifiers.
    while True:
        new = re.sub(r"\s*[\(\[]\s*[^)\]]*\s*[\)\]]\s*$", "", cleaned).strip()
        if new == cleaned:
            break
        cleaned = new

    # "first and last name" questions: drop a single middle token in a
    # three-token alphabetic name (e.g. "Rishi Raj Kapoor" -> "Rishi Kapoor").
    if re.search(r"\bfirst\s+and\s+last\s+name\b", question or "", flags=re.IGNORECASE):
        parts = cleaned.split()
        if len(parts) == 3 and all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", part) for part in parts):
            cleaned = f"{parts[0]} {parts[2]}"

    # For actors/actresses asked for a "full name", prefer the shorter form
    # only when that exact shorter form appears in the collected evidence
    # (e.g. evidence says "Rishi Kapoor" while the model wrote "Rishi Raj
    # Kapoor").  This avoids stripping genuine middle names from scientists.
    if (
        re.search(r"\b(actor|actress)\b", question or "", flags=re.IGNORECASE)
        and re.search(r"\bfull\s+name\b", question or "", flags=re.IGNORECASE)
    ):
        parts = cleaned.split()
        if len(parts) == 3 and all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", part) for part in parts):
            particle = parts[1].lower() in {
                "van", "von", "de", "der", "la", "le", "bin", "al", "da", "di",
            }
            short = f"{parts[0]} {parts[2]}"
            if not particle and short.lower() in (evidence or "").lower():
                cleaned = short

    # Title questions: prefer the title before a descriptive subtitle colon
    # (e.g. "The Legend of Scarface: A Blackfeet Indian Tale").
    if ":" in cleaned and re.search(
        r"\b(title|book|novel|film|movie|song|album|story)\b",
        question or "",
        flags=re.IGNORECASE,
    ):
        head, tail = cleaned.split(":", 1)
        head, tail = head.strip(), tail.strip()
        if len(head) >= 3 and re.match(r"^(a|an|the|part|volume|book)\b", tail, flags=re.IGNORECASE):
            cleaned = head

    return cleaned.strip().strip("\"'` ").strip()
