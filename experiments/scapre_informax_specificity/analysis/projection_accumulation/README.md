# ScaPre projection accumulation

## Formal Confuse5 outcome

The five-seed formal run completed with all qualification, isolation, and
evaluation-integrity checks passing. Projection accumulation improved mean
preserve and overall accuracy, but mean unlearn accuracy increased by 0.5
percentage points and the cats group showed a repeated regression. It therefore
failed the preregistered directional requirements and is treated as a valid
negative result. No COCO evaluation was launched, and this modification is not
being tuned or continued.

The curated formal artifacts and independent retrieval validation are in
[`formal_results/`](formal_results/README.md).

This experiment replaces only the earlier per-concept Informax accumulation
mask with a deterministic squared-cosine projection mask. The production
editor remains byte-unchanged. Official Informax still runs at every aggregate
and accumulation call, and its accumulation result is discarded only at the
`for_mat1 * row_w_c` multiplication.

On the local Mac, the repository-safe static check is:

```bash
experiments/scapre_informax_specificity/analysis/projection_accumulation/validate_static.sh
```

On the GPU server:

```bash
cd /home/tslin/Documents/jupyter_data/anLi/machine_unlearning
conda activate MU
experiments/scapre_informax_specificity/analysis/projection_accumulation/run_projection_confuse5.sh
```

The launcher intentionally fails unless the implementation is committed and
`git status --porcelain` is empty. It uses the active `MU` environment's
`python`; it does not activate Conda or hard-code an interpreter.

The detached job first completes seed-20260820 qualification. It starts the
five-seed Confuse5 evaluation only if every qualification and isolation gate
passes. It never calls the COCO runner.

```bash
experiments/scapre_informax_specificity/analysis/projection_accumulation/run_projection_confuse5.sh --status
experiments/scapre_informax_specificity/analysis/projection_accumulation/run_projection_confuse5.sh --package
```

COCO is a separately launched, project-defined secondary general-generation
safeguard. It is not a reproduction of the ScaPre paper COCO protocol and is
never started automatically.

Only after manual review of Confuse5:

```bash
experiments/scapre_informax_specificity/analysis/projection_accumulation/run_projection_coco.sh --first-1k
experiments/scapre_informax_specificity/analysis/projection_accumulation/run_projection_coco.sh --status
experiments/scapre_informax_specificity/analysis/projection_accumulation/run_projection_coco.sh --package
```

First-1k FID is labeled descriptive screening only. It never continues to
10,000 prompts. A later first-10k run is another explicit launch and requires a
completed first-1k run:

```bash
experiments/scapre_informax_specificity/analysis/projection_accumulation/run_projection_coco.sh --first-10k
```

The COCO worker uses edit seed `20260820`, the exact SD1.5 snapshot, float16,
PNDM 50 steps, CFG 7.5, 512x512, and the established ordered OCE COCO prompt and
seed asset. It compares official and projection checkpoints with CLIP and FID.
Existing SD1.4 references are rejected. If the matching SD1.5 reference is not
complete, the worker builds and registers a new exact-fingerprint reference.
That registration creates reviewable repository changes under
`orthogonal-concept-erasure/experiments/evaluation_references/`; those reference
artifacts must be reviewed and committed before a later clean-worktree
first-10k launch. Neither COCO mode is run as part of Confuse5.
