# Optional candidate-generation contract

No paid API is required by this experiment. Generate requests with `prepare-prompts`, send each JSONL request to any available language model, and save one response per line using this shape:

```json
{
  "request_id": "cat__visual_appearance",
  "concept": "cat",
  "facet_id": "visual_appearance",
  "descriptions": ["First sentence.", "Second sentence."]
}
```

The importer does not trust generated text. Every imported sentence is rechecked by `validate-text`, including banned terms, length, sentence count, and near-duplicate rules. The full configuration expects exactly 15 imported candidates for every concept/facet group.
