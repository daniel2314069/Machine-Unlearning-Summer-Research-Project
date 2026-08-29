# ScaPre direct-cos2 accumulation

This is a post-V1 exploratory follow-up. It is not part of the original
projection-accumulation preregistration and must not be described as a
confirmatory continuation of that failed formal experiment.

The sole scientific change is the earlier per-concept accumulation weight:

```text
projection_accumulation V1:
    sigmoid(zscore(projection_score) / 0.7) ** 8

projection_accumulation_direct_cos2:
    projection_score
```

The exact score is reused without changing epsilon placement:

```text
((W @ (concept_vecs[k] - empty_vec)) ** 2)
/
((row_norm_sq + 1e-8) * (difference_norm_sq + 1e-8))
```

There is no z-score, sigmoid, temperature, power, max normalization, energy
matching, learned parameter, or sweep in the treatment alpha. The frozen V1
transform is still computed only for descriptive diagnostics. Official
Informax runs normally at every accumulation and aggregate call, preserving
its RNG stream; aggregate `row_w_max` remains official and unintercepted. The
production editor remains byte-unchanged.

The detached worker performs, in order:

1. Descriptive analysis of all 320 records from the completed V1 seed-20260820
   diagnostics. This does not select or tune the formula.
2. Seed-20260820 official/direct-cos2 qualification, including weighted
   contribution norms, matrix-level `V` norms, checkpoint parameter delta,
   RNG/isolation checks, and bitwise `row_w_max` equality.
3. Five-seed Confuse5 generation only if qualification passes.
4. Fail-closed aggregation, then stop. It never starts COCO.

On the GPU server:

```bash
cd /home/tslin/Documents/jupyter_data/anLi/machine_unlearning
conda activate MU
experiments/scapre_informax_specificity/analysis/projection_accumulation_direct_cos2/run_direct_cos2_confuse5.sh
```

An optional server-environment and Python-syntax check does not launch work:

```bash
experiments/scapre_informax_specificity/analysis/projection_accumulation_direct_cos2/run_direct_cos2_confuse5.sh --preflight
```

Status and packaging:

```bash
experiments/scapre_informax_specificity/analysis/projection_accumulation_direct_cos2/run_direct_cos2_confuse5.sh --status
experiments/scapre_informax_specificity/analysis/projection_accumulation_direct_cos2/run_direct_cos2_confuse5.sh --package
```

Packaging has no `jq` dependency, excludes generated images and checkpoints,
and leaves the original server outputs intact.

This implementation does not provide or invoke a COCO runner. If the
Confuse5 directional conditions pass, the worker still stops for manual review.
COCO would be a separately authorized general-generation safeguard, not an
independent confirmation of the adaptively proposed Confuse5 variant.
