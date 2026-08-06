# Shared evaluation references

This directory is the repository-level registry for reusable evaluation
baselines. Evaluation references belong here rather than inside one edited-model
experiment.

Before generating an Original SD baseline, an evaluator should query
`registry.json` through `reference_registry.resolve_reference`. A reference may
be reused only when its complete protocol identity matches exactly, including:

- base model;
- prompt-source hash and ordered subset;
- scheduler, denoising steps and guidance scale;
- resolution and generation dtype;
- seed source and prompt count;
- metric model and implementation.

The registry deliberately rejects a familiar name with a different protocol
fingerprint. An entry with status `building` is not reusable.

Current OCE preservation reference ids:

- `sd14_mscoco30k_first1000_pndm50_cfg7p5_512_bf16`
- `sd14_mscoco30k_first10000_pndm50_cfg7p5_512_bf16`

They retain the Original CLIP baseline, compact FID Inception statistics,
protocol manifest and exact ordered prompt manifest. Generated PNGs and
per-image Inception features are not retained after successful evaluation.

For a new edited model, use `--stop-after-first1000` as a medium-cost screening
gate. The evaluator writes 1k metrics and retains resumable edited images. Run
the same command without the flag only when the model should continue to the
paper first-10k protocol.

Inspect the registry with the required project environment:

```bash
conda run -n py310 python experiments/evaluation_references/reference_registry.py list
```
