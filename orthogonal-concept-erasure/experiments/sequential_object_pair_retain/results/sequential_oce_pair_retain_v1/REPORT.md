# Sequential OCE Pair-Erasure: Previous-Target Retain Results

## Technical summary

Overall assessment: **share with caveats**. The run and all requested aggregates pass independent validation, but two Stage-1 edits (`automobile` and `bird`) did not erase their targets successfully. Results are descriptive for the fixed five-pair schedule and do not establish a general causal effect beyond it.

1. **The second edit sometimes weakens the first erasure, but not generally.** Under normal sequential OCE, first-target accuracy increased in 3 of 10 orders, was unchanged in 4, and decreased in 3. The mean raw change was **+0.60 percentage points (pp)**, the median was **0 pp**, and the largest increase was **+8.5 pp** for `bird→automobile`.
2. **Order matters for some concepts and is negligible for others.** Under normal sequential OCE, the largest final-accuracy differences between erasing a concept first versus second were **10.0 pp for bird** and **7.5 pp for automobile**. `deer↔dog` was order-invariant at the displayed 0.5 pp resolution; `frog↔ship` differed by 2.5 pp for each concept.
3. **Adding the previous target to the second retain set is a selective, not reliable, improvement.** Relative to normal Stage 2, first-target final accuracy was lower in 5 orders, tied in 2, and higher in 3. The mean difference was **−1.05 pp** and the median was **−0.50 pp**. The result is dominated by an **−8.5 pp** change for `bird→automobile`, whose bird Stage-1 edit had already failed.
4. **The modification has a small but non-zero second-target cost and essentially no aggregate remaining-eight cost.** Second-target accuracy changed by **+0.35 pp on average** (higher is worse for an erased target): 5 orders worsened, 4 tied, and 1 improved. Remaining-eight mean accuracy changed by **−0.019 pp on average**, with 7 exact ties and a range from −0.125 to +0.063 pp.

The appropriate conclusion is therefore: retain-previous is a **pair-dependent safeguard**, not a universally better replacement for normal sequential OCE.

## Original baseline is effectively at the classifier ceiling

Original SD v1.4 reached **100% accuracy on eight classes** and **99.5% on automobile and truck**, for a ten-class mean of **99.9%**. The low edited-target accuracies therefore do not arise from a weak Original classifier baseline.

## Stage-1 validity is strong for eight orders and fails for two

Stage 1 is the standard single-concept OCE condition. Eight first edits reached target accuracy at or below 7.5%. `automobile→bird` started with automobile target accuracy of 90.0%, and `bird→automobile` started with bird target accuracy of 57.0%; neither represents a clean first erasure.

| Order | First target | Acc_e ↓ | Acc_s ↑ | H_o ↑ |
|---|---:|---:|---:|---:|
| airplane→cat | airplane | 1.5% | 100.000% | 99.244% |
| cat→airplane | cat | 0.0% | 100.000% | 100.000% |
| automobile→bird | automobile | 90.0% | 100.000% | 18.182% |
| bird→automobile | bird | 57.0% | 100.000% | 60.140% |
| deer→dog | deer | 0.0% | 99.889% | 99.944% |
| dog→deer | dog | 0.5% | 100.000% | 99.749% |
| frog→ship | frog | 5.0% | 100.000% | 97.436% |
| ship→frog | ship | 5.5% | 99.944% | 97.146% |
| horse→truck | horse | 7.5% | 100.000% | 96.104% |
| truck→horse | truck | 7.5% | 100.000% | 96.104% |

The fixed `automobile↔bird` pair remains in every primary table; it is not removed or replaced. Its failure only limits how strongly its sequential changes can be interpreted as retention of a completed erasure.

## Normal Stage 2 causes order-specific first-target changes

Positive values below mean the first erased target became more classifiable after the second edit, so the first erasure weakened. Negative values mean target accuracy decreased further.

| Order | Normal Stage-2 change from Stage 1 | Retain-previous change from Stage 1 | Retain minus normal, first target | Retain minus normal, second target | Retain minus normal, remaining-eight mean |
|---|---:|---:|---:|---:|---:|
| airplane→cat | +3.0 pp | +2.0 pp | −1.0 pp | 0.0 pp | 0.000 pp |
| cat→airplane | 0.0 pp | 0.0 pp | 0.0 pp | 0.0 pp | 0.000 pp |
| automobile→bird | 0.0 pp | +1.5 pp | +1.5 pp | +0.5 pp | 0.000 pp |
| bird→automobile | +8.5 pp | 0.0 pp | −8.5 pp | 0.0 pp | 0.000 pp |
| deer→dog | 0.0 pp | +0.5 pp | +0.5 pp | +1.0 pp | 0.000 pp |
| dog→deer | 0.0 pp | 0.0 pp | 0.0 pp | 0.0 pp | −0.125 pp |
| frog→ship | −2.0 pp | −3.0 pp | −1.0 pp | −2.0 pp | −0.125 pp |
| ship→frog | +0.5 pp | −1.0 pp | −1.5 pp | +1.0 pp | +0.063 pp |
| horse→truck | −2.0 pp | −1.5 pp | +0.5 pp | +1.5 pp | 0.000 pp |
| truck→horse | −2.0 pp | −3.0 pp | −1.0 pp | +1.5 pp | 0.000 pp |

The normal second edit causes three observed rebounds: airplane (+3.0 pp), bird (+8.5 pp), and ship (+0.5 pp). It leaves four first targets unchanged and further lowers three. This answers the first research question with **sometimes**, not **systematically**.

## Retain-previous improves five orders but does not dominate normal Stage 2

For the first target, retain-previous is better than normal Stage 2 in:

- `airplane→cat`: −1.0 pp
- `bird→automobile`: −8.5 pp
- `frog→ship`: −1.0 pp
- `ship→frog`: −1.5 pp
- `truck→horse`: −1.0 pp

It ties for `cat→airplane` and `dog→deer`, and is worse for `automobile→bird` (+1.5 pp), `deer→dog` (+0.5 pp), and `horse→truck` (+0.5 pp).

The largest improvement is attached to the failed bird Stage-1 edit. Restricting interpretation—not pair selection—to the eight orders with Stage-1 target accuracy at or below 7.5%, retain-previous is better in 4, tied in 2, and worse in 2, with a mean first-target difference of −0.438 pp versus normal Stage 2. That is modest evidence of benefit, not a robust universal effect.

For the second target, retain-previous is worse in five orders by 0.5–1.5 pp, ties in four, and improves `frog→ship` by 2.0 pp. It does not prevent the second erasure wholesale, but it is not cost-free.

## Final erasure efficacy differs between A→B and B→A for selected concepts

The table gives each concept's final target accuracy when erased first minus its final target accuracy when erased second. It is a direct raw difference, not a newly named metric. Positive means final accuracy was higher when erased first; negative means it was higher when erased second.

| Pair | Concept | Normal sequential | Retain previous |
|---|---|---:|---:|
| airplane↔cat | airplane | +2.5 pp | +1.5 pp |
| airplane↔cat | cat | 0.0 pp | 0.0 pp |
| automobile↔bird | automobile | −7.5 pp | −6.0 pp |
| automobile↔bird | bird | +10.0 pp | +1.0 pp |
| deer↔dog | deer | 0.0 pp | +0.5 pp |
| deer↔dog | dog | 0.0 pp | −1.0 pp |
| frog↔ship | frog | −2.5 pp | −4.5 pp |
| frog↔ship | ship | −2.5 pp | −2.0 pp |
| horse↔truck | horse | 0.0 pp | −1.0 pp |
| horse↔truck | truck | +0.5 pp | −2.0 pp |

The clearest order dependence occurs in the pair whose individual Stage-1 edits fail, so the 10.0 pp and 7.5 pp differences should be reported but not generalized. Outside that pair, normal sequential differences range from 0 to 2.5 pp.

## Exact per-order target and remaining-eight accuracies

All values are CLIP ten-class classification accuracies from 200 images per class. Lower is better for the first and second targets; higher is better for the remaining-eight mean.

| Order | First after Stage 1 | First after normal Stage 2 | First after retain Stage 2 | Second normal | Second retain | Remaining 8 normal | Remaining 8 retain |
|---|---:|---:|---:|---:|---:|---:|---:|
| airplane→cat | 1.5% | 4.5% | 3.5% | 0.0% | 0.0% | 100.000% | 100.000% |
| cat→airplane | 0.0% | 0.0% | 0.0% | 2.0% | 2.0% | 100.000% | 100.000% |
| automobile→bird | 90.0% | 90.0% | 91.5% | 55.5% | 56.0% | 100.000% | 100.000% |
| bird→automobile | 57.0% | 65.5% | 57.0% | 97.5% | 97.5% | 100.000% | 100.000% |
| deer→dog | 0.0% | 0.0% | 0.5% | 0.5% | 1.5% | 99.875% | 99.875% |
| dog→deer | 0.5% | 0.5% | 0.5% | 0.0% | 0.0% | 99.938% | 99.813% |
| frog→ship | 5.0% | 3.0% | 2.0% | 8.5% | 6.5% | 100.000% | 99.875% |
| ship→frog | 5.5% | 6.0% | 4.5% | 5.5% | 6.5% | 99.875% | 99.938% |
| horse→truck | 7.5% | 5.5% | 6.0% | 5.0% | 6.5% | 100.000% | 100.000% |
| truck→horse | 7.5% | 5.5% | 4.5% | 5.5% | 7.0% | 100.000% | 100.000% |

Except for the failed `automobile↔bird` pair, final second-target accuracies are at most 8.5% under normal Stage 2 and at most 7.0% under retain-previous. Remaining-eight preservation is 99.813–100% in every displayed condition.

## Scope, protocol, and metric definitions

- Base model: `CompVis/stable-diffusion-v1-4`
- Generation prompt: `a photo of the {class}`
- Classes: all ten CIFAR-10 object classes
- Fixed unordered pairs: airplane↔cat, automobile↔bird, deer↔dog, frog↔ship, horse↔truck
- Ordered sequences: both directions for every pair
- Images: 200 per class per evaluated cell
- Seeds: integers 42–241, fixed before generation and shared across conditions
- Sampler: PNDM
- Inference: 50 steps, CFG 7.5, 512×512, bfloat16 generation
- Evaluator: `openai/clip-vit-base-patch32`, argmax over ten prompts of the form `photo of a {class}`
- Stage-1 metrics: Acc_e, mean nine-class Acc_s, and paper H_o
- Stage-2 reporting: both target accuracies and every remaining-eight class accuracy; no multi-target H_o

Target accuracy is lower-is-better. First-target change is final accuracy after Stage 2 minus accuracy immediately after Stage 1. Retain-versus-normal differences are retain-previous accuracy minus normal Stage-2 accuracy.

## Independent validation and data integrity

The included standard-library Ruby audit re-read all saved predictions without loading a model. It verified:

- 62,000 prediction rows and 310 evaluator cells
- exactly 200 rows, seeds 42–241, and sample indices 0–199 per cell
- no duplicate `(group, checkpoint, concept, sample_index, seed)` keys
- correct prompt and expected-label mapping for every row
- maximum ten-class probability-sum error of `2.3519e-7`
- exact reproduction of all 310 `per_class_results.csv` rows
- exact reproduction of all 10 `summary.csv` rows
- exact reproduction of all Stage-1 Acc_e, Acc_s, and H_o values
- all 310 per-cell predictions/metrics SHA-256 markers
- both Stage-2 variants sharing the same Stage-1 parent checkpoint for each order
- successful formal-image cleanup markers for all evaluated cells

The runner's own `final_validation.json` independently reports status `complete`, 62,000 generated images, 62,000 predictions, 310 evaluators, 10 ordered pairs, shared seeds, shared Stage-1 parents, successful prediction recomputation, completed cleanup, 140 qualitative raw images, and 20 contact sheets.

## Qualitative evidence

The fixed qualitative archive contains every pair, both orders, both targets, and seeds 42 and 43 across Original, Stage 1, normal Stage 2, and retain-previous Stage 2. Its 140 raw images and 20 contact sheets match the precommitted manifest and archive hash.

Visual inspection is consistent with strong erasure for most classes and conspicuous failures for automobile and bird. Some normal-versus-retain samples are nearly unchanged while others diverge. Because there are only two qualitative seeds per target, these images are illustrative and are not used to replace the formal evaluator.

## Limitations and robustness boundaries

- This is a fixed-pair descriptive experiment, not a statistical claim over a broader concept population.
- Accuracy resolution is 0.5 pp per class because each class has 200 images.
- Two Stage-1 target erasures failed; the most favorable retain-previous result occurs in one of them.
- Seeds are matched across conditions, but the report does not add a paired significance test because the requested protocol centers raw accuracies and raw differences.
- The retained qualitative images are deterministic fixed examples, not a representative quantitative sample.
- The portable canonical `artifact.json` was created, but this Mac has no Node executable, so the official HTML renderer could not be run locally. This Markdown report is the readable fallback; the HTML omission does not affect numerical validation.

## Recommended next step

Do not make retain-previous the default from this run alone. If confirmation is warranted, repeat the same fixed pair schedule and unchanged hyperparameters with additional independently fixed seed sets or full repetitions. Predeclare the checks as:

1. whether the five observed first-target improvements recur;
2. whether improvement remains after excluding—not replacing—the two failed Stage-1 cases from interpretation;
3. whether the observed 0.5–1.5 pp second-target penalties recur;
4. whether remaining-eight preservation stays effectively unchanged.

## Further questions

- Does `bird→automobile` still benefit when the first bird edit succeeds?
- Which 0–2.5 pp order differences outside `automobile↔bird` replicate?
- Are the small second-target penalties stable across independent seed sets?
