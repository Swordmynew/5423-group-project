#!/usr/bin/env python3
"""Extract text from a local PDF or an HTTP(S) PDF.

This is the offline helper for the pdf-extractor skill.  The agent itself uses
``document_reader(..., extract_pdf=true)``; this script is for inspecting a file
while developing or debugging.

Usage:
    python scripts/extract_pdf.py report.pdf
    python scripts/extract_pdf.py https://example.org/report.pdf --max-chars 8000
    python scripts/extract_pdf.py report.pdf --out report.txt
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path


def read_source(source: str, timeout: int = 30, max_bytes: int = 50 * 1024 * 1024) -> bytes:
    if source.lower().startswith(("http://", "https://")):
        request = urllib.request.Request(
            source,
            headers={"User-Agent": "Mozilla/5.0 (compatible; COMP5423PDFHelper/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(max_bytes)
    else:
        data = Path(source).expanduser().read_bytes()

    if not data.startswith(b"%PDF"):
        raise SystemExit(f"Not a PDF file: {source}")
    return data


def extract_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit("pypdf is required. Install it with `pip install pypdf`.") from exc

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - malformed PDFs
            text = f"[page {number} extraction failed: {exc}]"
        pages.append(f"\n===== PAGE {number} =====\n{text.strip()}")
    return "\n".join(pages).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Local PDF path or HTTP(S) URL.")
    parser.add_argument("--max-chars", type=int, default=0, help="Truncate output (0 = all).")
    parser.add_argument("--start", type=int, default=0, help="Start offset in characters.")
    parser.add_argument("--out", type=Path, default=None, help="Write to a UTF-8 file instead of stdout.")
    args = parser.parse_args()

    text = extract_text(read_source(args.source))
    if args.start:
        text = text[args.start :]
    if args.max_chars and args.max_chars > 0:
        text = text[: args.max_chars]

    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {len(text):,} characters to {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(text)


if __name__ == "__main__":
    main()
