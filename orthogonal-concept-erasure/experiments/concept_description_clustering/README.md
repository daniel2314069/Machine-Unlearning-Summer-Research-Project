# Original-SD1.4 concept-description clustering

This is an isolated, resumable experiment. It does not import `oce.py`, call OCE, load an edited state dict, create an edited `W`, or change any existing OCE script. Generation, contextual embedding extraction, and layer projection all load `CompVis/stable-diffusion-v1-4` from the original checkpoint.

The experiment asks whether name-free descriptions cluster by concept, whether they cluster by facet instead, and whether explicit concept-name prototypes lie near the corresponding name-free description centroids.

## Isolation guarantees

- Original unedited SD 1.4 only.
- No anchors and no concept-erasure operation.
- No FID and no edited-image comparison.
- Generation uses the candidate exactly as written, without a suffix.
- Embedding extraction appends one configured suffix and audits its final token.
- K-means is fitted only on accepted name-free descriptions; prototypes are held out.
- Every layer analysis uses the same original W0 projection matrix for all descriptions.
- The exact checkpoint revision plus original projection names/shapes are compared across generation and embedding. A dtype-specific in-memory W0 fingerprint is also recorded at each stage for audit; its bytes are not compared across bfloat16 and float32 loads.

## Environment

All Python commands use the repository-required Conda environment:

```bash
cd /home/daniel1012/projects/machine_unlearning/orthogonal-concept-erasure/experiments/concept_description_clustering
conda run -n py310 pip install -r requirements.txt
```

`scripts/run_py310.sh` invokes Python through `conda run -n py310` and also exposes the CUDA-13 runtime path used by the current workstation.

## Configurations

- `configs/full.json`: 10 concepts × 10 facets × 15 candidates, retain exactly 10 accepted per group.
- `configs/pilot.json`: cat/dog/rabbit × 3 facets × 5 candidates, formal three-seed thresholds.
- `configs/smoke.json`: the same 45 curated pilot candidates, one Stage-1 seed only. It validates mechanics and is not a formal inclusion run.
- `configs/syntax_independent_4x50.json`: cat/dog/fox/bear, 50 accepted descriptions per concept, three independently collected sources, lexical baselines, word shuffle, and prototype percentiles.
- `configs/syntax_independent_4x100.json`: the corresponding formal 100-description-per-concept condition.

JSON inheritance is resolved before execution and saved as `resolved_config.json` in each output directory. Banned names and synonyms are configured per concept.

## Candidate data without a mandatory LLM API

The repository contains no required paid LLM integration. A complete curated 45-row pilot is included at `data/pilot_candidates.jsonl`.

To prepare the 100 full-generation requests (10 concepts × 10 facets, requesting 15 sentences each):

```bash
./scripts/run_py310.sh -m concept_clustering.cli prepare-prompts \
  --config configs/full.json \
  --output-file data/full_generation_requests.jsonl
```

After obtaining external responses in the documented JSONL response shape:

```bash
./scripts/run_py310.sh -m concept_clustering.cli import-candidates \
  --responses data/full_generation_responses.jsonl \
  --output-file data/full_candidates.jsonl
```

The importer never trusts external text; `validate-text` applies all configured rules and saves every failure reason.

## Syntax-independent follow-up

This condition intentionally has no bundled fake "multi-source" candidate data. Source labels are evidence, so responses must actually be collected independently from a human author and two distinct model/prompting systems. No paid API is required by the repository.

Create 120 source-separated requests (4 concepts × 10 facets × 3 sources):

```bash
./scripts/run_py310.sh -m concept_clustering.cli prepare-prompts \
  --config configs/syntax_independent_4x100.json \
  --output-file data/syntax_independent_generation_requests.jsonl
```

The generated requests are included, and the response/provenance contract is documented in `templates/syntax_independent_collection.md`. After collecting the responses:

```bash
./scripts/run_py310.sh -m concept_clustering.cli import-candidates \
  --responses data/syntax_independent_generation_responses.jsonl \
  --output-file data/syntax_independent_candidates.jsonl \
  --source independently_collected
```

Run the 4 × 50 pilot:

```bash
./scripts/run_py310.sh -m concept_clustering.cli run \
  --config configs/syntax_independent_4x50.json \
  --candidates data/syntax_independent_candidates.jsonl \
  --output outputs/syntax_independent_4x50 \
  --projection to_v
```

Run the 4 × 100 formal condition with the same 600-candidate input after generation review has enough accepted rows for the stricter source quotas:

```bash
./scripts/run_py310.sh -m concept_clustering.cli run \
  --config configs/syntax_independent_4x100.json \
  --candidates data/syntax_independent_candidates.jsonl \
  --output outputs/syntax_independent_4x100 \
  --projection to_v
```

The validator rejects repeated trigrams, repeated three-content-word openings, repeated syntax-family labels within a concept/facet, source imbalance, banned labels, and near duplicates. Final selection also enforces the configured per-source quota in every concept/facet instead of merely reporting source imbalance afterward.

The analysis adds three unsupervised TF-IDF conditions (word unigram, word unigram+bigram, and character 3–5 gram), five-fold supervised TF-IDF and fixed-readout linear probes reported separately, and a deterministic word-order shuffle that preserves each case-folded word bag. Prototype rows include empirical, leave-one-out, and 1,000-split bootstrap distance percentiles; a high percentile means the explicit name is peripheral relative to its name-free descriptions.

## Codex-diverse unattended experiment

The `codex_diverse` condition is a separate, honestly single-source exploratory corpus for cat, dog, fox, and bear. Its deterministic builder creates 250 candidates per concept across the same ten facets, applies strict lexical/syntactic diversity validation, then uses five-fold out-of-fold TF-IDF predictions to prioritize 12 difficult candidates per concept/facet for original-SD1.4 generation validation. It does not claim independent human or multi-model provenance.

Launch the isolated 20-hour job in the background:

```bash
./scripts/codex_diverse_job.sh launch outputs/codex_diverse_overnight
```

After a reboot or recoverable interruption, resume cached work with a fresh wall-clock budget (set `MAX_HOURS` to override 20):

```bash
./scripts/codex_diverse_job.sh resume outputs/codex_diverse_overnight
```

Read the live state and heartbeat:

```bash
./scripts/codex_diverse_job.sh status outputs/codex_diverse_overnight
```

Rerun only clustering/reporting from cached accepted descriptions and embeddings:

```bash
./scripts/codex_diverse_job.sh cluster-only outputs/codex_diverse_overnight
```

The runner checkpoints generation after every image, writes `state.json` and `heartbeat.json`, logs subprocess output to `stage_subprocess.log`, and writes its final or timeout summary to `OVERNIGHT_REPORT.md`. It first targets 200 balanced accepted descriptions. If every facet has ten eligible rows and at least six hours remain, it upgrades the cached corpus to 400 without regenerating images.

## Pipeline

1. `validate-text`: schema, label/facet membership, deterministic `langdetect` English scoring, one sentence, 8–20 words, banned terms, negation/comparison, informativeness, exact and near duplicates, and balanced candidate counts.
2. `validate-generation`: original SD 1.4 Stage 1, independent CLIP ensemble classification, then Stage 2 only for Stage-1 passes. Images and scores are cached per candidate/seed.
3. `finalize`: combine automatic and optional manual decisions, rank eligible candidates by margin, require the exact per-group count, and fail on shortages.
4. `extract`: audit tokenization; extract fixed-readout, natural-last-token, and held-out prototype representations; project through every original `attn2.to_v` layer.
5. `cluster`: 20 k-means seeds with `n_init=100`, concept/facet metrics, assignments, prototype analysis, confusion matrices, and plots.
6. `report`: write a technical Markdown report from saved evidence.

## Exact smoke-test command

This runs text validation and one original-model generation seed for cat, dog, and rabbit across three facets with five candidates per group:

```bash
./scripts/run_py310.sh -m concept_clustering.cli smoke \
  --config configs/smoke.json \
  --candidates data/pilot_candidates.jsonl \
  --output outputs/smoke
```

Because it uses one seed, it stops after Stage 1 and is not the formal 2-of-3 inclusion test.

## Exact full-run command

For the completed, visually reviewed experiment in this repository, use the
final 1,500-row candidate revision `data/full_candidates_v14.jsonl`:

```bash
./scripts/run_py310.sh -m concept_clustering.cli run \
  --config configs/full.json \
  --candidates data/full_candidates_v14.jsonl \
  --output outputs/full_to_v \
  --projection to_v
```

This is the automatic path. If any concept/facet has fewer than ten eligible candidates, it writes `facet_shortages.csv` and stops instead of borrowing from another facet.

For optional manual review, run `validate-text` and `validate-generation` separately, edit only the `manual_decision` and `manual_notes` columns in `manual_review.csv`, then run `finalize`, `extract`, `cluster`, and `report`.

If review rescues a visually correct Stage-1 image that CLIP mislabeled, set
`manual_decision=accept` and rerun `validate-generation --stage 2 --resume`.
The candidate is then promoted only to Stage 2 so seeds 43 and 44 are still
generated; the single reviewed image is not treated as sufficient formal evidence.

## Exact resume command

Generation is resumable and skips cached candidate/seed images and score rows:

```bash
./scripts/run_py310.sh -m concept_clustering.cli validate-generation \
  --config configs/full.json \
  --output outputs/full_to_v \
  --stage all \
  --resume
```

## Exact cached-clustering command

This requires existing `accepted_descriptions.jsonl`, `raw_text_embeddings.pt`, and `layer_embeddings.pt`. It does not load SD or regenerate images:

```bash
./scripts/run_py310.sh -m concept_clustering.cli analyze \
  --config configs/full.json \
  --output outputs/full_to_v
```

To run the optional original `to_k` condition, pass `--projection to_k`; `to_v` remains the default. Use a separate output directory for each projection condition so its `layer_embeddings.pt` and derived metrics remain unambiguous.

## Decision logic

The CLIP backend ensembles:

- `a photo of a {concept}`
- `an image of a {concept}`
- `a clear picture of a {concept}`

It records all class logits/probabilities, target rank, target score, target-versus-best-other margin, normalized entropy, low-confidence status, and high-confidence wrong-class status. Automatic acceptance requires the intended class to be top-1 for at least two of three formal seeds plus the configured probability and mean-margin gates, with no low-confidence or high-confidence-wrong seed. Low-confidence and ambiguous cases are not forced into a class.

CLIP ViT-B/32 is independent of the SD 1.4 CLIP ViT-L/14 text encoder, but it is still a closed-set classifier. Contact sheets and manual review are therefore part of the audit path.

## Fixed contextual readout

The primary suffix is exactly:

```text
 This sentence describes the concept
```

The code tokenizes each complete prompt without truncation, selects the final non-special token, verifies that its ID is identical across every accepted description and prototype, verifies that it decodes to `concept`, and saves all token IDs/tokens in `tokenization_audit.csv`. Any mismatch or truncation aborts extraction.

The main representation is the final-layer contextual hidden state at that token. It is 768-dimensional and L2-normalized. Natural last-token extraction is saved as a separate control.

## Outputs

The pipeline writes at least:

- `candidate_descriptions.jsonl`
- `candidate_text_validation.csv`
- `candidate_text_validation_failures.jsonl`
- `candidate_diversity_audit.csv`
- `candidate_diversity_summary.csv`
- `generation_validation.csv`
- `candidate_generation_decisions.csv`
- `accepted_descriptions.jsonl`
- `manual_review.csv`
- `tokenization_audit.csv`
- `word_shuffle_audit.csv` (when enabled)
- `raw_text_embeddings.pt`
- `layer_embeddings.pt`
- `clustering_metrics.csv`
- `clustering_assignments.csv`
- `prototype_metrics.csv`
- `lexical_baseline_clustering_metrics.csv` (when enabled)
- `lexical_baseline_assignments.csv` (when enabled)
- `baseline_classification_metrics.csv` (when enabled)
- `tfidf_top_features.csv` (when enabled)
- `facet_confounding_metrics.csv`
- `layer_metrics.csv`
- `concept_by_facet_error_table.csv`
- `confusion_matrices/`
- `plots/`
- `final_report.md`

## Expected runtime and storage

- Smoke Stage 1: about 3–5 minutes and roughly 20–40 MB.
- Full generation: approximately 1,500 Stage-1 images plus two additional images for Stage-1 passes. At about four seconds per image on the current RTX 5060, expect roughly 3–5 hours depending on the pass rate; iterative review-driven revisions add time.
- Full clustering: `20 runs × n_init=100 × raw/control/layer conditions` is intentionally expensive on CPU; expect roughly 0.5–4 additional hours depending on scikit-learn and CPU threading.
- The completed `outputs/full_to_v` occupies about 2.3 GB, including 4,206 active generated PNGs, contact sheets, and revision archives. The embedding caches are roughly 55 MB.

## Manual review

Manual review is optional for clear automatic cases and recommended for all `borderline` rows, obvious malformed/toy/partial-animal images, and high-impact rejections. Allowed values are `accept`, `reject`, and `unset`. A manual decision never changes the original image or classifier score; it only affects final eligibility.

## Assumptions

- The locally cached `CompVis/stable-diffusion-v1-4` is the intended original checkpoint.
- The repository's normal generation settings are the existing `generate_object.py` defaults: 50 steps, CFG 7.5, 512×512, pipeline-default scheduler, and bfloat16 generation.
- The current environment can load `openai/clip-vit-base-patch32`; if not cached, the first classifier run needs model download access.
- Full candidate text is supplied later through the included optional request/import workflow because no existing mandatory LLM API was found.
