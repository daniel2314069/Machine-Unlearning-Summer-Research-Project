# OCE failure image qualification

This experiment follows the completed zero-generation qualification audit. It
does not alter `oce.py`, introduce a head-local OCE method, or repair the
determinant correction.

Before generation it records per-head source-coordinate leakage, official
rotation orthogonality residuals, singular-value ranges, and three
pre-registered determinant-correction cases. Direction 1 then compares
Original SD with official OCE using canonical target prompts followed by a
GenEval-style two-object detector smoke. Direction 3 compares two legal
numerical-null-space SVD realizations while every other edited layer is held
identical.

On tslin:

```bash
cd orthogonal-concept-erasure
conda activate MU
bash experiments/oce_failure_image_qualification/run_server.sh --allow-network
```

Status:

```bash
bash experiments/oce_failure_image_qualification/status_server.sh
```

The resumable project output is stored under
`experiments/oce_failure_image_qualification/outputs/qualification_v1/`.
After successful completion, only the report, manifests, diagnostics, and
metric tables are packaged under
`/home/tslin/Documents/jupyter_data/anLi/tmp/`. The final `scp` remains manual.

To retain a small, fixed-rule visual sanity-check set without copying all 440
generated images, package 50 review images and then remove only the project
image directories:

```bash
bash experiments/oce_failure_image_qualification/package_review_images.sh
bash experiments/oce_failure_image_qualification/cleanup_server_images.sh
```

The cleanup script refuses to run until the review-image archive exists. It
preserves reports, metrics, manifests, and checkpoints.
