---
name: multi-clue-decomposer
description: Decompose a multi-constraint question into independent clues, search the most discriminative clue first, and track which candidate satisfies each clue. Use when a question contains several facts, dates, relations, or nested conditions.
---

# Multi-Clue Decomposer

## When to use

Use when the question is not one lookup: several clauses joined by "and",
"who", "whose", "after", or semicolons; multiple dates, places, or roles; or an
answer that must satisfy all of several constraints. Do not use it for a short
factoid.

## Procedure

1. **Fix the answer type** — person, organisation, place, date, number, title.
2. **Build a clue table**:

   | # | Clue (as stated) | Type | First query | Status |
   |---|---|---|---|---|
   | 1 | born in the 1910s | date | `born 1914 chemist` | open |
   | 2 | discovered a compound for a disease | relation | `compound disease discovery` | open |

3. **Search the most discriminative clue first**: rare names, exact phrases,
   unusual relations, distinctive institutions. Avoid broad fields and common
   words.
4. **One clue, one query**: run `unified_retrieval`; open the best 1–2 documents
   with `document_reader` using `focus`; mark the clue supported, contradicted,
   or unknown. If two query variants fail, mark unknown and move on.
5. **Track candidates**:

   | Candidate | Clue 1 | Clue 2 | Evidence doc IDs |
   |---|---|---|---|
   | A | yes | yes | 123, 456 |

   Eliminate candidates that contradict a confirmed clue.
6. **Verify**: the final candidate must satisfy every clue; check the name and
   answer type with `text_searcher` inside the strongest document, and with a
   second independent document if possible.

## Query patterns

- `"<rare name>" <role>`
- `"<year>" "<event phrase>"`
- `"<candidate>" "<other entity>"`
- exact phrase -> `text_searcher`, not a broad retrieval query
- alternative spelling / historical name if the first attempt fails

## Anti-patterns

- Searching the entire question verbatim.
- Combining unrelated clues into one query.
- Treating a 200-character snippet as proof.
- Repeating the same query after a weak result.
- Choosing a famous answer instead of a documented one.

Finish with `FINAL ANSWER: <answer>`.
