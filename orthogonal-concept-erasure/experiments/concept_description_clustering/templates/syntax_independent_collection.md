# Syntax-independent candidate collection protocol

Collect the three source slots independently. Do not show one source the text
written by another source. A source may be a human author or a distinct language
model/prompting system, but its identity and model/version should be recorded in
the response metadata.

Each request in `data/syntax_independent_generation_requests.jsonl` expects a
JSON response with this shape:

```json
{
  "request_id": "llm_source_a__cat__visual_appearance",
  "source": "llm_source_a",
  "concept": "cat",
  "facet_id": "visual_appearance",
  "source_metadata": {
    "author_or_model": "record the actual author/model",
    "model_version": "record an immutable version when available",
    "collection_date": "YYYY-MM-DD"
  },
  "descriptions": [
    {
      "description": "One name-free English sentence goes here.",
      "syntax_family": "short human-readable construction label"
    }
  ]
}
```

Requirements:

- Do not reuse responses, prompts, or examples across source slots.
- Do not build descriptions by crossing fixed concept-specific subjects with
  shared facet predicates.
- Within a concept/facet/source request, use five genuinely different subject
  and sentence constructions.
- `syntax_family` labels must be distinct within each concept/facet across all
  three sources. They are audit metadata, not true labels used by clustering.
- Preserve rejected text and source provenance. Do not rewrite failures in
  place; create a new candidate revision.
- The deterministic validator rejects repeated trigrams, repeated lexical
  openings, repeated syntax-family labels, banned concepts, and near duplicates
  across the complete candidate file.

The main analysis remains generation-filtered. Retain the full text-valid
population so selection effects can be quantified separately from the final
accepted population.
