# OCE Target–Anchor Correspondence Diagnostic

## Technical summary

- Phase A and Phase B use Stable Diffusion 1.4, the same 20 seeds (42–61), 50 denoising steps, guidance 7.5, 512×512 output, and all 16 `attn2.to_v` layers.
- The vector-wise checkpoint uses the paper's paired objective; the subspace checkpoint reproduces the current upstream `oce.py` objective. No `Y_tilde`, new loss, clustering, sequential editing, or alternate erasure method is present.
- The automatic screening label uses only directional evidence: edited target similarity must decrease, edited own-anchor similarity must increase, and low-variance image incidence must stay at or below 20%. It does not remove mappings automatically.
- Directional screening labels: passed `cat_to_dog, church_to_building, french_horn_to_trumpet`; did not pass `airplane_to_sky, golf_ball_to_tennis_ball`. These labels require visual/user review before an N=5 set is selected.

## 1. Single-pair screening

CLIP similarities are cosine similarities from `openai/clip-vit-base-patch32` using the text template `a photo of a {concept}`. “Own-anchor top-1” is evaluated against the pair's target and anchor labels. A pass is descriptive, not a positive conclusion or an automatic inclusion decision.

| Mapping | Original target | Edited target | Edited own anchor | Target Δ | Anchor Δ | Own-anchor top-1 | Directional label |
|---|---:|---:|---:|---:|---:|---:|---|
| cat → dog | 0.2811 | 0.2196 | 0.2774 | -0.0615 | +0.0502 | 100.0% | pass |
| airplane → sky | 0.2675 | 0.2710 | 0.2253 | +0.0035 | +0.0270 | 5.0% | does not pass |
| church → building | 0.2904 | 0.2584 | 0.2594 | -0.0320 | +0.0286 | 45.0% | pass |
| golf ball → tennis ball | 0.3142 | 0.3235 | 0.2627 | +0.0094 | +0.0099 | 0.0% | does not pass |
| French horn → trumpet | 0.3338 | 0.3242 | 0.3124 | -0.0096 | +0.0037 | 0.0% | pass |

Per-image results: [initial_per_image_metrics.csv](metrics/initial_per_image_metrics.csv)  
Machine-readable summary: [single_pair_screening_summary.json](metrics/single_pair_screening_summary.json)

### Manual grid review

- **cat → dog:** Original images consistently depict cats. Both edited methods consistently produce dog-dominant animals. Single-vector images occasionally retain pointed-ear or large-eye features that could be read as mildly feline; single-subspace has a few unusual ear/head artifacts, most visibly around seeds 49 and 57. Neither method shows systematic background collapse, blank output, or broad image-quality failure.
- **airplane → sky:** Edited images remain airplane-dominant across the grid. Sky or cloud backgrounds are sometimes more prominent, but the target object is not removed; several outputs are grayscale or have silhouette/geometry artifacts. The visual result agrees with the failed directional target-erasure check.
- **church → building:** Several edited outputs move from church interiors/exteriors toward generic office, apartment, or plain building facades, while many rows remain visibly church-like or retain church architecture. This is a partial and seed-dependent shift without systematic generation collapse.
- **golf ball → tennis ball:** Most edited outputs remain dimpled golf balls. A few become green spheres, but they generally lack the seam and felt texture of a tennis ball. The grid does not support reliable golf-ball erasure or tennis-ball replacement.
- **french horn → trumpet:** Many edited outputs lengthen the instrument or add trumpet-like bells and tubing, but a substantial fraction retain circular French-horn geometry or become brass-instrument hybrids. This is visually a partial shift; it does not establish clean trumpet replacement, consistent with the 0% two-label trumpet top-1 rate.

Structured visual-review notes: [manual_visual_review.json](metrics/manual_visual_review.json)


### Cat → dog smoke test

| Method | Mean cat similarity | Mean dog similarity | Dog top-1 rate |
|---|---:|---:|---:|
| Original SD | 0.2811 | 0.2271 | 0.0% |
| Single vector-wise | 0.2196 | 0.2774 | 100.0% |
| Single subspace | 0.2081 | 0.2739 | 100.0% |

![Cat to dog smoke grid](grids/smoke_cat_to_dog.png)

Per-seed cells are `cat similarity / dog similarity / top-1 label`.

| Seed | Original | Single vector-wise | Single subspace |
|---:|---|---|---|
| 42 | 0.2869/0.2283/cat | 0.2207/0.2829/dog | 0.2085/0.2812/dog |
| 43 | 0.2796/0.2289/cat | 0.2252/0.2865/dog | 0.1992/0.2685/dog |
| 44 | 0.2952/0.2392/cat | 0.1962/0.2626/dog | 0.1914/0.2626/dog |
| 45 | 0.2806/0.2310/cat | 0.2138/0.2769/dog | 0.2096/0.2787/dog |
| 46 | 0.2889/0.2356/cat | 0.2116/0.2770/dog | 0.2088/0.2708/dog |
| 47 | 0.2807/0.2260/cat | 0.2250/0.2805/dog | 0.1952/0.2705/dog |
| 48 | 0.2726/0.2151/cat | 0.2160/0.2774/dog | 0.2110/0.2786/dog |
| 49 | 0.2817/0.2226/cat | 0.2392/0.2894/dog | 0.2463/0.2804/dog |
| 50 | 0.2847/0.2317/cat | 0.2057/0.2620/dog | 0.1999/0.2702/dog |
| 51 | 0.2705/0.2190/cat | 0.2034/0.2693/dog | 0.1951/0.2573/dog |
| 52 | 0.2893/0.2304/cat | 0.2237/0.2804/dog | 0.1967/0.2653/dog |
| 53 | 0.2649/0.2116/cat | 0.2165/0.2765/dog | 0.2093/0.2784/dog |
| 54 | 0.2851/0.2298/cat | 0.2494/0.2908/dog | 0.2082/0.2757/dog |
| 55 | 0.2823/0.2274/cat | 0.2283/0.2953/dog | 0.2009/0.2705/dog |
| 56 | 0.2709/0.2241/cat | 0.2148/0.2683/dog | 0.1875/0.2562/dog |
| 57 | 0.2701/0.2226/cat | 0.2031/0.2660/dog | 0.2209/0.2752/dog |
| 58 | 0.2817/0.2298/cat | 0.2378/0.2817/dog | 0.2123/0.2776/dog |
| 59 | 0.2933/0.2378/cat | 0.2204/0.2729/dog | 0.2301/0.2859/dog |
| 60 | 0.2866/0.2321/cat | 0.2258/0.2814/dog | 0.2086/0.2793/dog |
| 61 | 0.2774/0.2197/cat | 0.2158/0.2693/dog | 0.2235/0.2947/dog |

### Screening image grids

#### airplane → sky

![airplane to sky screening grid](grids/screening_airplane_to_sky.png)

#### church → building

![church to building screening grid](grids/screening_church_to_building.png)

#### golf ball → tennis ball

![golf ball to tennis ball screening grid](grids/screening_golf_ball_to_tennis_ball.png)

#### French horn → trumpet

![French horn to trumpet screening grid](grids/screening_french_horn_to_trumpet.png)

## 2. N = 2 joint vector-wise

Pending. This phase has not been executed in the initial smoke/screening run.

## 3. N = 2 joint subspace

Pending. This phase has not been executed in the initial smoke/screening run.

## 4. N = 5 joint vector-wise

Gated. The recorded directional rule yields 3/5 eligible candidates (`cat_to_dog, church_to_building, french_horn_to_trumpet`). This is fewer than five, so no N=5 checkpoint or image was created and no replacement mapping was added.

## 5. N = 5 joint subspace

Gated by the same candidate-count check as the vector-wise N=5 condition. See [joint_n5_gate.json](metrics/joint_n5_gate.json). No N=5 generation was run.

## 6. Control-set preservation

Pending until the joint checkpoints are evaluated. Preflight verified that `horse`, `ship`, `truck`, `frog`, and `deer` do not overlap the configured target or anchor sets.

## Scope, definitions, and resolved settings

- Target similarity: cosine similarity between a generated image and its target evaluation text.
- Own-anchor similarity: cosine similarity between a generated image and its paired anchor evaluation text.
- Screening candidate set: exactly the target and own anchor for that single-pair mapping.
- Cg: `/home/daniel1012/projects/machine_unlearning/orthogonal-concept-erasure/Cg.pt`, SHA-256 `e9ad216caa06097f1cb3d3edd02c56f98aabca1490b8d6e892a586f5d1b912ae`, `403727` valid tokens.
- OCE scales: erase `2000.0`, global retain `10.0`, local retain `0.0`, lambda `10.0`.
- Object prompt expansion: `true`; each bare pair plus `image/photo/portrait/picture/painting of` paired forms.
- Full inputs and parameters: [resolved_config.json](resolved_config.json), [target_anchor_pairs.csv](inputs/target_anchor_pairs.csv), [prompts.csv](inputs/prompts.csv), [seeds.csv](inputs/seeds.csv).
- Artifact QA: [artifact_validation.json](metrics/artifact_validation.json).

## Methodology and validation notes

The image comparisons are paired by prompt and seed. Raw cosine similarities—not two-label softmax probabilities—are stored. Low RGB variance is only a mechanical collapse alarm; semantic collapse and cat–dog hybridization require inspection of the saved grids. The current sample has no confidence interval and is intended as a diagnostic screen, not a population estimate.

## Limitations, uncertainty, and robustness checks

- A two-label top-1 result can improve even when both absolute similarities are poor; raw similarities are therefore reported beside top-1 rates.
- Single-pair screening does not measure cross-anchor confusion because there is only one anchor. N=2/N=5 matrices are required for correspondence claims.
- CLIP similarity does not establish object identity, visual quality, or absence of hybrids.
- The automatic directional label uses zero-crossing deltas and a low-variance alarm. It is intentionally not an effect-size threshold or an automatic exclusion rule.

## Recommended next steps

1. Review all five screening grids and retain the numeric screening decisions as annotations rather than automatic exclusions.
2. Run N=2 joint vector-wise and joint subspace with `cat → dog` and `airplane → sky`.
3. Select the N=5 mapping list explicitly; do not add replacement mappings if fewer than five candidates are accepted.
4. Run the control-set comparison only after the joint checkpoints exist.

## Further questions

- Do visually acceptable images agree with the directional CLIP label for every candidate?
- Does N=2 reduce own-anchor correspondence relative to the corresponding single-pair models?
- Does feature-level diagonal structure predict the image-level confusion matrix?
