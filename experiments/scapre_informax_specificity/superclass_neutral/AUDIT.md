# Superclass-neutral pre-run audit

## Research scope

This is the final ablation on the existing ScaPre Informax specificity line. It
changes only the five Informax negative/reference base embeddings per target:

- `official`: five noisy copies of the empty-prompt embedding.
- `superclass_neutral`: five noisy copies of one fixed superclass embedding.

The MI estimator, sample counts, noise, median threshold, binary states,
standardization, soft weighting, max aggregation, edit solver, model,
generation protocol, classifier, and metrics remain unchanged.

## Repository implementation

- Core editor: `scapre/edit/erase_scale.py`
- Informax estimator: `_compute_mi_softmask_emptyneg` and the reference-vector
  equivalent `_compute_mi_softmask_matchedneg`
- Integration: `edit_model.compute_informax`
- Official reference: `empty_vec = blank_emb[0, 1, :]`
- Per target and call: 5 positive + 5 negative pseudo-samples
- Intervention: `--informax-negative-mode superclass-neutral` plus
  `--informax-superclass-config`
- Default: `official`; existing `matched-retain` behavior remains available.

`superclass-neutral` passes exactly one superclass vector to the existing
reference-vector estimator. Because indexing one vector five times only replaces
the negative base, random draw counts and tensor shapes remain the same.

## Fixed mappings

| Group | Targets | Superclass |
| --- | --- | --- |
| dogs | golden retriever; labrador retriever | dog |
| cats | tabby; tiger cat | cat |
| fruits | orange; lemon | fruit |
| boats | yawl; lifeboat | boat |
| balls | soccer ball; volleyball | ball |

The mapping follows the actual repository Confuse5 targets. In particular,
`speedboat` is a retain class; the boat targets are `yawl` and `lifeboat`.
No retain label is used as an Informax reference.

## Baseline reuse and qualitative set

Formal official and matched-retain score rows are imported from the verified
five-seed robustness run. They are not regenerated. The formal run produces
15,000 new superclass-neutral evaluation images (5 seeds × 3,000).

Seed `20260820` follows the same legacy globally seeded editor path as its
imported official baseline. Seeds `20260821–20260824` use the already validated
Informax-only RNG wrapper while preserving the fixed legacy stream for every
non-Informax draw. This distinction is necessary for true seed pairing and is
recorded per seed in `controlled_ablation_check.json`.

The qualitative set is fixed before results: both targets and one retain from
each group, using protocol sample indices 0 and 1, for 30 images per variant.
Superclass-neutral images are copied from its formal evaluation; deleted
official and matched-retain examples are regenerated from their seed-20260820
checkpoints with the identical prompt, generation seed, scheduler, steps, CFG,
resolution, and safety checker. If those lightweight comparison checkpoints no
longer exist, only the two seed-20260820 checkpoints are deterministically
recreated; no baseline score evaluation is rerun.
