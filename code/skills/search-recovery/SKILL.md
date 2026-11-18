---
name: search-recovery
description: Recover from empty, irrelevant, or contradictory retrieval by diagnosing the failure, reformulating the query, relaxing one constraint, using exact-phrase or metadata search, and switching hypotheses. Use after a weak search result.
---

# Search Recovery

## When to use

- `unified_retrieval` returns nothing or only irrelevant pages;
- the same query keeps returning the same weak documents;
- snippets mention the topic but not the requested relation;
- two documents disagree;
- a name may be translated, abbreviated, or historical.

Do not use it when the first results already answer the question.

## Diagnose first

| Symptom | Likely cause | First move |
|---|---|---|
| Zero results | query too long, misspelling, stopword-heavy | shorten to 2–4 rare terms; spelling variants |
| Right topic, wrong relation | broad overview pages | add the relation verb (`designed`, `born`, `founded`) |
| One dominant page | a long overview document | open it; `text_searcher` inside it |
| Wrong period | ambiguous dates | add a decade or `metadata_filter` dates |
| Ambiguous entity | common name | add role, country, or field |
| Contradictory snippets | truncation or different entities | open both documents |

## Recovery ladder

Work one step at a time and inspect the result.

1. **Shorten and sharpen**: keep proper nouns, numbers, unusual nouns; remove
   question words and generic descriptions.
2. **Change vocabulary**: synonyms, British/American spelling, historical/local
   names, expanded/abbreviated names, source language.
3. **Drop one constraint**: search the remaining two clues, then test the
   dropped clue on the candidates. Never drop the rare relation first.
4. **Exact phrase**: pass a short quoted phrase to `text_searcher` in global
   mode (under ~8 words).
5. **Filter the catalogue**: `metadata_filter` by domain, title, date range, or
   length, then open the results.
6. **Search inside the best document**: `document_reader` then
   `text_searcher(doc_id=...)` with the candidate name, date, or relation.
7. **Switch hypothesis**: if a named entity is consistently contradicted, stop
   forcing it and re-read the clues.

## Budget

- At most 2–3 recovery attempts per clue.
- Two reformulations with no relevant result -> mark the clue unknown.
- If a candidate satisfies all but one unverifiable clue, verify the candidate
  independently and answer.
- Never fabricate a document ID or claim a page says something it does not.

Finish with `FINAL ANSWER: <answer>`.
