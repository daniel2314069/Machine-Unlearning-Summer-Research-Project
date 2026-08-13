# AFR matrix QA and conditional balls smoke

This directory contains an experiment-local implementation of anchor-fixed
residual contraction and its mandatory pure-projection ablation. It does not
change production `oce.py` or any existing OCE checkpoint.

The three matrix variants are:

- `C_objective_faithful`: the frozen float64 objective-faithful OCE baseline;
- `F_pure_residual_projection`: `T=D=I-R_e` at `alpha=1`;
- `G_full_afr`: `T=P*D`, where `P` is the frozen-S-optimal orthogonal
  compensation subject to fixing the anchor subspace pointwise.

Targets, matched anchors, prompt expansion, K0, local retain concepts, scales,
and the 16 edited layers are loaded from the frozen Confuse5 protocol. In
particular, balls retains its registry mapping `soccer ball -> basketball` and
`volleyball -> baseball`.

## GPU execution

Run from the `orthogonal-concept-erasure` repository root in the GPU server's
project environment:

```bash
conda activate MU

# Optional fast algebra-only QA (no model load)
python experiments/confuse5_single_vs_joint/afr/runner.py synthetic

# Matrix QA only; never generates images
python experiments/confuse5_single_vs_joint/afr/runner.py matrix

# Full requested workflow. The balls smoke starts only after AFR-GO.
python experiments/confuse5_single_vs_joint/afr/runner.py all --skip-existing
```

`all` first computes all 48 dogs/fruits/balls layer cases in float64. It stops
without image generation on `AFR-M0` or `AFR-M1`. On `AFR-GO`, it writes three
new balls-only checkpoints in this experiment's own output namespace and runs
exactly 1,800 new images: 500 target/preservation images plus 100 anchor-prompt
images per C/F/G model. It does not generate a new Original baseline and does
not rerun existing Single or released Joint results.

The output root defaults to:

```text
experiments/confuse5_single_vs_joint/afr/outputs/afr_balls_smoke_v1/
```

Important files are:

```text
matrix/results_c_f_g.csv
matrix/gate.json
REPORT.md
checkpoints/<variant>/weights.safetensors
checkpoints/<variant>/metadata.json
balls_smoke/resolved_plan.json
balls_smoke/evaluations/*.json
balls_smoke/anchor_lpips.json
balls_smoke/summary.json
```

The output namespace is git-ignored because it can contain checkpoints and
images. To transfer the compact results back, include `REPORT.md`, the matrix
CSV/gate, smoke summary, LPIPS JSON, resolved plan, evaluator shards, and
checkpoint metadata; do not include checkpoints or PNGs unless explicitly
needed.

## Safety and failure behavior

- The full matrix stage refuses to run on macOS or without CUDA.
- All audit algebra is float64; saved edited weights are cast to float32 for
  the existing checkpoint loader.
- `alpha=0` is an explicit no-op; the requested run is fixed at `alpha=1`.
- F/G leakage and constrained anchor-feature error must be numerical zero.
- G must not have worse frozen-S distortion than F, and its compensation must
  be measurably nontrivial in multiple layers.
- Image/evaluator dependencies and completed reference artifacts are checked
  before the first new PNG is generated.
- Existing checkpoint outputs are never overwritten. `--skip-existing` only
  reuses artifacts after fingerprint and SHA-256 validation.

The anchor guarantee means exact preservation of the constrained anchor
features at each edited layer. It does not claim that anchor-prompt image
generation is invariant; that is why the conditional smoke evaluates anchor
accuracy/probability and fixed-seed Original-vs-edited LPIPS.

## Recorded result

The completed compact GPU result is tracked under
[`results/afr_balls_smoke_v1/`](results/afr_balls_smoke_v1/). Its matrix gate
passed (`AFR-GO`), but the image smoke classified the editor as `AFR-I0` because
F/G did not improve target erasure or matched-anchor semantic movement.
