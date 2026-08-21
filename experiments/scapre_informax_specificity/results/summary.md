# ScaPre Informax Specificity Experiment

## Technical summary

The formal paired run passes the frozen `SUPPORTED` rule, but only narrowly on
the preservation threshold. Replacing empty-prompt Informax negatives with five
balanced-as-possible same-group retain negatives raised Preserve Accuracy from
40.61% to 41.72% (`+1.11` percentage points), lowered residual target Unlearn
Accuracy from 19.75% to 17.42% (`-2.33` points; lower is better), and raised
Overall Accuracy from 53.93% to 55.44% (`+1.51` points). Preservation improved
in four groups and tied in fruits. This supports the tested replacement under
the pre-registered descriptive rule; it does not establish that every retain
concept benefits or that the result reproduces the paper's unavailable full
Table 7 seed asset.

## A. Audit

At workspace commit `d7a48beb7796c666dc949245ea10627d16086055`, the audited repository uses five target pseudo-samples against five empty-prompt pseudo-samples. It applies the repository's median binarization, empirical MI, channel z-score, sigmoid temperature, power transform, and channel-wise maximum aggregation. The complete audit, including file paths, functions, equations, and data flow, is in `experiments/scapre_informax_specificity/AUDIT.md`.

The public repository does not contain the complete paper Confuse5 seed asset. This run is therefore labeled **project-established Confuse5 reconstruction derived from public ScaPre assets**, not an exact reproduction of paper Table 7.

## B. Modification

The only algorithmic intervention is the negative base-vector source inside `scapre/edit/erase_scale.py`. The existing `_compute_mi_softmask_emptyneg` remains the `official` default; `_compute_mi_softmask_matchedneg` changes only the negative bases to the three listed same-group retain embeddings, selected by the `compute_informax` dispatcher inside `edit_model`. Both modes use exactly five negatives, identical Gaussian-noise draw shapes, and the same downstream Informax and ScaPre code. Five negatives are assigned round-robin as 2/2/1 in declared retain order. `controlled_ablation_check.json` verifies that the two normalized edit commands differ only in the intervention and variant-specific artifact paths.

## C. Reproducibility

- Profile: `formal`
- Workspace commit: `d7a48beb7796c666dc949245ea10627d16086055`
- Working tree dirty at launch: `True` (required caveat; the archive preserves and hash-pins the editor and all experiment sources, and their critical hashes match the current commit, but the package cannot identify unrelated dirty paths)
- Base model: `runwayml/stable-diffusion-v1-5`
- Resolved model revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
- Protocol SHA-256: `ab8ced3be82b1d8c6896eb24d32419bb12e8cf3e22d8ed0c46f92f4a5e8f0f2e`
- Images per concept: `120`
- Prompt template: `an image of a {concept}`
- Seed sources: `{"project-derived:same-group-retain-seed-reuse": 1200, "public-repo:imagenet-15.csv": 1800}`
- Generation: base-model scheduler, 50 steps, CFG 7.5, 512x512, float16
- Classifier: `torchvision ResNet50_Weights.DEFAULT` (resolved asset `IMAGENET1K_V2`), top-1, repository substring label mapping
- Informax edit seed: `20260820` for both variants

Validation checks performed after download:

- archive SHA-256 verified as `35fde6c41d150146ef3022e830a772a547afb3234d731fa4acf03f9c9d09e4d1`;
- worker completion marker and exit code `0` verified;
- both score files contain exactly 3,000 rows: 1,200 target and 1,800 retain rows, with 120 images for each of 25 concepts;
- all 3,000 `(group, role, concept, sample_index, prompt, seed, seed_source)` keys match exactly across the protocol, official, and matched-retain files, with no duplicate keys;
- aggregate and five group metrics were independently recomputed from the raw `correct` field and match the emitted tables;
- normalized edit commands match after removing only the intended Informax mode/config, diagnostics path, and variant output path;
- the Stable Diffusion safety checker returned black images 223 times for official and 225 times for matched-retain. The checker and its behavior were held fixed, and matched-retain did not gain an advantage from fewer safety-checker substitutions, but the high substitution rate limits interpretation of absolute classifier accuracy.

## D. Main table

| Variant | Unlearn Acc ↓ | Preserve Acc ↑ | Overall Acc ↑ |
| --- | --- | --- | --- |
| official | 19.75 | 40.61 | 53.93 |
| matched_retain | 17.42 | 41.72 | 55.44 |
| delta | -2.33 | 1.11 | 1.51 |

## E. Per-group table

| Group | Off U | Match U | Δ U | Off P | Match P | Δ P | Off O | Match O | Δ O |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dogs | 16.67 | 11.67 | -5.00 | 46.11 | 47.50 | 1.39 | 59.37 | 61.78 | 2.41 |
| cats | 27.92 | 30.42 | 2.50 | 19.44 | 21.39 | 1.94 | 30.63 | 32.72 | 2.09 |
| fruits | 10.00 | 7.92 | -2.08 | 18.61 | 18.61 | 0.00 | 30.84 | 30.96 | 0.12 |
| boats | 20.83 | 20.00 | -0.83 | 91.94 | 93.33 | 1.39 | 85.08 | 86.15 | 1.08 |
| balls | 23.33 | 17.08 | -6.25 | 26.94 | 27.78 | 0.83 | 39.87 | 41.61 | 1.74 |

## F. Per-concept table

| Group | Role | Concept | Official | Matched | Delta |
| --- | --- | --- | --- | --- | --- |
| dogs | target | golden retriever | 13.33 | 6.67 | -6.67 |
| dogs | target | labrador retriever | 20.00 | 16.67 | -3.33 |
| dogs | retain | german shepherd | 45.83 | 49.17 | 3.33 |
| dogs | retain | chesapeake bay retriever | 10.00 | 13.33 | 3.33 |
| dogs | retain | pug | 82.50 | 80.00 | -2.50 |
| cats | target | tabby | 13.33 | 10.83 | -2.50 |
| cats | target | tiger cat | 42.50 | 50.00 | 7.50 |
| cats | retain | persian cat | 19.17 | 20.00 | 0.83 |
| cats | retain | siamese cat | 38.33 | 44.17 | 5.83 |
| cats | retain | egyptian cat | 0.83 | 0.00 | -0.83 |
| fruits | target | orange | 8.33 | 7.50 | -0.83 |
| fruits | target | lemon | 11.67 | 8.33 | -3.33 |
| fruits | retain | pomegranate | 17.50 | 20.00 | 2.50 |
| fruits | retain | fig | 15.00 | 13.33 | -1.67 |
| fruits | retain | granny smith | 23.33 | 22.50 | -0.83 |
| boats | target | yawl | 36.67 | 34.17 | -2.50 |
| boats | target | lifeboat | 5.00 | 5.83 | 0.83 |
| boats | retain | speedboat | 98.33 | 99.17 | 0.83 |
| boats | retain | catamaran | 88.33 | 90.00 | 1.67 |
| boats | retain | schooner | 89.17 | 90.83 | 1.67 |
| balls | target | soccer ball | 36.67 | 24.17 | -12.50 |
| balls | target | volleyball | 10.00 | 10.00 | 0.00 |
| balls | retain | tennis ball | 19.17 | 19.17 | 0.00 |
| balls | retain | rugby ball | 27.50 | 32.50 | 5.00 |
| balls | retain | ping-pong ball | 34.17 | 31.67 | -2.50 |

## G. Informax diagnostic

These are mechanism diagnostics only and are not success metrics. Across `320` matched layer/projection/target records, mean Spearman alpha correlation is `0.0644`. Mean official-vs-matched top-channel overlap is `0.4022` at 1%, `0.1030` at 5%, and `0.1749` at 10%. Mean raw channel MI changed from `0.6847` to `0.3796`. Raw MI/alpha tensors remain in the downloaded archive under `diagnostics/*.pt`; exact indices are in `results/top_channels.json`.

## H. Final judgment

**SUPPORTED**

matched-retain negatives consistently improve similar-concept preservation without materially weakening target erasure.

This judgment follows the frozen rule: the `+1.11`-point aggregate Preserve
gain reaches the 1.0-point minimum, four of five groups have a positive
Preserve delta, and aggregate Unlearn Accuracy improves rather than exceeding
the 2.0-point degradation margin. The effect is not uniform at concept level:
9 of 15 retains improve, 1 ties, and 5 worsen; the fruits group ties, and cats'
target Unlearn Accuracy worsens by 2.50 points. No confidence interval or
hypothesis test was pre-registered, so `SUPPORTED` is a threshold-based result,
not a claim of statistical significance. Given the reconstructed seed asset,
safety-checker substitutions, and dirty-at-launch marker, the result is ready
to share only with these caveats and should not be described as an exact paper
Table 7 reproduction.
