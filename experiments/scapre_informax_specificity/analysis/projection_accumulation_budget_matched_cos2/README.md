# ScaPre budget-matched direct-cos2 accumulation

This is the final exploratory geometric accumulation modification. It removes
the strength confound observed in direct cos2 and tests only row-wise
allocation.

For each target concept and each accumulation matrix, the experiment computes
the same direct-cos2 score as the previous variant:

```text
r = ((W @ (concept_vecs[k] - empty_vec)) ** 2)
    / ((row_norm_sq + 1e-8) * (difference_norm_sq + 1e-8))
```

It then fixes the total contribution strength to official:

```text
C_official = for_mat1 * row_w_c_official
C_geo      = for_mat1 * r.view(-1, 1)
lambda     = ||C_official||_F / (||C_geo||_F + 1e-8)
C_new      = lambda * C_geo
```

There is no z-score, sigmoid, power, max normalization, lambda clamp, sweep,
fallback, retain data, or concept-specific rule. Official Informax executes at
every accumulation and aggregate call. Its accumulation alpha is used only for
`||C_official||_F`; aggregate `row_w_max` remains official and unintercepted.
The production editor remains byte-unchanged.

The pre-implementation code-path and isolation audit is recorded in
[`IMPLEMENTATION_AUDIT.md`](IMPLEMENTATION_AUDIT.md).

Qualification uses edit seed `20260820` and fails closed unless all 320
concept/matrix records are finite, non-degenerate, and satisfy
`||C_new||_F ~= ||C_official||_F` at the frozen integrity tolerance
`rtol=1e-5, atol=1e-7`. These tolerances are validation tolerances, not treatment
parameters. Qualification passing automatically proceeds to the established
five-seed Confuse5 treatment generation and then stops. It never starts COCO.

On the GPU server:

```bash
cd /home/tslin/Documents/jupyter_data/anLi/machine_unlearning
conda activate MU
experiments/scapre_informax_specificity/analysis/projection_accumulation_budget_matched_cos2/run_budget_matched_cos2_confuse5.sh
```

Preflight, status, and packaging:

```bash
experiments/scapre_informax_specificity/analysis/projection_accumulation_budget_matched_cos2/run_budget_matched_cos2_confuse5.sh --preflight
experiments/scapre_informax_specificity/analysis/projection_accumulation_budget_matched_cos2/run_budget_matched_cos2_confuse5.sh --status
experiments/scapre_informax_specificity/analysis/projection_accumulation_budget_matched_cos2/run_budget_matched_cos2_confuse5.sh --package
```

Both preflight and launch run a lightweight synthetic MU-environment test of
the budget arithmetic, RNG non-consumption, historical hashes, report builder,
and result manifest before any model is loaded.

Packaging has no `jq` dependency. It excludes generated images and regenerable
checkpoints, leaves server image outputs intact, and writes the archive under
`/home/tslin/Documents/jupyter_data/anLi/tmp`. `retrieval_validation.json` is
created only after the archive is downloaded and independently checked on the
Mac; it is not fabricated by the server before retrieval.

V1 and direct-cos2 formal means are included as hash-validated historical
descriptive comparisons. The only formal baseline is official. Whether the
automatic directions pass or fail, the run stops after Confuse5 for manual
semantic-pattern review. No fourth cos2 variant is authorized by this runner.
