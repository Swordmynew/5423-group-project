# Recovery strategies: quick reference

This sheet is a compact checklist for the search-recovery skill. It is original
teaching material, not a copy of any external skill.

## Query surgery checklist

1. Is the query longer than 8 content words? Cut it.
2. Does it contain question words (`who`, `which`, `what`, `why`)? Remove them.
3. Does it contain a date? Keep it only if the date is distinctive.
4. Is there a proper noun? Keep its exact spelling first, then try variants.
5. Is there a relation word (`designed`, `founded`, `discovered`, `succeeded`)?
   Keep one strong relation verb.
6. Would a domain expert use a different technical term? Try that term.

## Recovery moves by corpus symptom

### Nothing returned

- Remove stopwords and boilerplate.
- Try the rarest term alone.
- Try a shorter substring (surname only, city only).
- Try a synonym or the local-language name.
- Search the exact phrase with `text_searcher`.

### Right topic, wrong relation

- Add one relation verb: `born`, `founded`, `designed`, `discovered`, `married`,
  `succeeded`, `subsidiary`, `alumnus`.
- Search both entities together: `"<entity A>" "<entity B>"`.
- Open the top overview page and use `text_searcher` inside it.

### Too many different entities

- Add a disambiguator: country, profession, century, organisation type.
- Use `metadata_filter` to restrict the domain or date.
- Add a second clue from the question.

### Contradictory evidence

- Prefer a primary or contemporaneous source over a listicle.
- Check whether the documents refer to different people with the same name.
- Look for the full relation, not just a name co-occurrence.
- If still unresolved, choose the candidate supported by more independent pages.

## Constraint priority

When dropping a constraint, drop it in this order:

1. generic descriptive phrases ("known for being kind");
2. broad time periods;
3. broad locations;
4. ordinary education/employment facts;
5. the rare relation or proper noun — never drop this first.

## Stop conditions

- Two reformulations with no relevant result -> mark the clue unknown.
- A candidate satisfies all but one clue, and that clue is unverifiable ->
  verify the candidate independently, then answer.
- Tool budget is nearly exhausted -> finalise with the best-supported candidate
  rather than starting a new line of inquiry.
