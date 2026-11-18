---
name: pdf-extractor
description: Read PDF documents when a corpus source is a PDF, its text is empty or garbled, or a report must be checked. Use document_reader with extract_pdf and text_searcher on the extracted document.
---

# PDF Extractor

## When to use

- the source URL ends in `.pdf` or points to a report/paper;
- `document_reader` returns empty or clearly broken text;
- extraction has broken words, page headers, or column order;
- the answer is in a table, footnote, or caption.

Prefer an HTML source if it already contains the fact.

## Procedure

1. **Open with PDF extraction**:

   ```
   document_reader(doc_id="<id>", extract_pdf=true, focus="<key words>", max_chars=3000)
   ```

   The extracted text is cached for later `text_searcher` calls on the same
   `doc_id`.

2. **If extraction fails**: the PDF may be scanned or the server may block
   downloads. Look for an HTML/transcript version or another document with the
   same fact. Do not guess what the PDF says.

3. **Search inside the extracted text**:

   ```
   text_searcher(query="<candidate name or phrase>", doc_id="<id>")
   ```

   Search the rarest token first (surname, year, model number).

4. **Repair artefacts mentally**:
   - hyphenated line breaks (`govern- ment`) -> search the unhyphenated word;
   - repeated headers/footers -> ignore;
   - multi-column order -> search the phrase, not the paragraph;
   - ligatures (`ﬁ`, `ﬂ`) -> search `fi`, `fl`;
   - digits separated from units -> search the digits.

5. **Verify** the fact against a second source when possible, then answer with
   `FINAL ANSWER: <answer>`.

## Offline helper

`scripts/extract_pdf.py` extracts a local PDF or URL outside the agent:

```
python scripts/extract_pdf.py report.pdf --max-chars 5000
python scripts/extract_pdf.py https://example.org/report.pdf --start 2000
```

## Limitations

- No OCR for scanned image PDFs.
- No table-structure recovery.
- Encrypted PDFs may not open.
- Respect source access rules and rate limits.
