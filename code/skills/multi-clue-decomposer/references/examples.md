# Worked examples (synthetic)

These examples are original and illustrative; they are not drawn from the
course dataset. They show how to turn a dense question into searchable clues.

## Example 1: an indirect biography question

Question: "A chemist born in the 1910s, who studied at a coastal university and
later discovered a compound used to treat a metal-storage disease, also competed
in a national chess tournament. What is the chemist's full name?"

Clue table:

| # | Clue | First query | Why this order |
|---|---|---|---|
| 1 | discovered a compound for a metal-storage disease | `compound metal storage disease` | most distinctive relation |
| 2 | competed in a national chess tournament | `chemist national chess tournament` | unusual hobby, narrows fast |
| 3 | studied at a coastal university | `coastal university chemistry 1930s` | broad; use only if needed |
| 4 | born in the 1910s | `born 1914 chemist` | verify at the end |

Search plan: run clue 1, open the best two documents, collect compound and
disease names; search clue 2 together with a candidate surname; then verify
clues 3-4 for the leading candidate. If clue 2 returns nothing, try "chess
championship" or the national federation's name.

## Example 2: a multi-constraint organisation question

Question: "A research institute founded in the 1950s in a capital city, named
after a scientist who refused a patent for a medical discovery, hosts a
telescope that made the first observation of a particular pulsar. Which institute
is it?"

Decomposition:

1. scientist refused a patent for a medical discovery -> identify the scientist;
2. institute named after that scientist, founded in the 1950s in a capital city;
3. telescope at that institute, first observation of a pulsar.

The first clue is the anchor because "refused a patent" is rarer than
"institute". Once the scientist is known, the institute name is a direct lookup,
and the pulsar clue verifies the identification.

## Example 3: when not to decompose

Question: "Who wrote the novel *The Silent Orchard*?"

One entity, one relation, no constraints. A single retrieval call is enough;
using the skill here would waste tool calls and tokens.
