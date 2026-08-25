# ScaPre Informax alpha-channel control: implementation audit

## Repository state at design time

- Branch: `main`
- Commit: `790ce34291c184abe5535f6ae89fe2e4e5432961`
- Worktree before implementation: clean
- Production editor: `scapre/edit/erase_scale.py`
- Established formal path: `experiments/scapre_informax_specificity/seed_robustness/worker.py` -> experiment-only Informax RNG wrapper -> production editor -> `evaluate_confuse5.py`

The formal server preflight records the launch commit, branch, start/end status,
source hashes, actual config, protocol hash, assets, Python executable, and Conda
environment again. The design-time SHA above is not substituted for run-time
provenance.

## Exact intervention point

For `to_v`, the official final aggregate alpha is produced by
`torch.max(torch.stack(row_ws_all, dim=-1), dim=-1).values` at production lines
541-547. It enters the closed-form edit at lines 615-623 as
`alpha_i = erase_scale * row_w_max[i]` and
`M_i = G_base + alpha_i * PiC`.

The same sequence is independently repeated for `to_k` at lines 659-666 and
730-738. Both projections therefore use the same kind of intervention point,
but have separate alpha vectors for every layer.

The implementation does not explicitly materialize a dense diagonal `B`.
`row_w_max` is the diagonal of that conceptual matrix, and the row loop applies
its coefficients without allocating `diag(alpha)`.

Per-concept alpha is also used earlier in the unchanged UCE accumulation
(`row_w_c` at lines 595-605 and 713-722). That is not the final concept-max
vector and is deliberately not controlled in this experiment. Changing it
would test a broader intervention than the requested channel-to-final-alpha
correspondence.

## Isolation mechanism

`alpha_control_runner.py` is an experiment-only `runpy` wrapper. It leaves the
production file byte-unchanged and intercepts only the two-argument reduction
whose caller is `edit_model`, whose input rank is three, whose reduction
dimension is `-1`, and whose occurrence is one of the expected 32 concept-max
reductions (16 `to_v`, then 16 `to_k`). Any count, shape, order, or caller drift
aborts the edit.

The wrapper first computes the unmodified max result, then replaces only its
`.values` tensor:

- `official`: unchanged;
- `constant_mean`: `full_like(alpha, alpha.mean())`, per matrix;
- `shuffled`: `index_select` by a CPU permutation keyed with SHA-256 over the
  preregistered salt, edit seed, projection, and zero-based layer;
- `identity_B`: `ones_like(alpha)`.

The formal shuffle salt is fixed in `config.json`. Two different, named salts
are smoke-only implementation checks and cannot be promoted by observed image
quality. A CPU-only `torch.Generator` is constructed for each permutation. It
does not read or advance global CPU/CUDA RNG state.

The wrapper reproduces the established five-seed RNG semantics exactly. Seed
`20260820` is the legacy run: Informax returns the normal global-stream draws.
For seeds `20260821`-`20260824`, every Informax call consumes the legacy global
draw and returns the corresponding tensor from an Informax-only generator keyed
by edit seed. Thus the reusable official result and each paired control share
the same seed-specific semantics, while entropy, UCE, prompts, generation RNG,
and all other random streams remain fixed.

## Official result reuse

The official five-seed image scores can be reused byte-for-byte from the
verified seed-robustness archive. Formal preflight requires:

- archive SHA-256 `df0874...3708`;
- protocol SHA-256 `ab8ced...0f2e`;
- exactly 3,000 unique rows per seed (1,200 target and 1,800 retain);
- the five pinned official `scores.csv` hashes;
- the canonical evaluator fingerprint `d149cd...cf98`;
- identical model/classifier assets to the current server assets manifest.

The historical editor/evaluator hashes are also pinned. Their exact diff to the
current checkout is pinned as `5a3d2b...41e3`; the audited diff only adds the
separate `superclass-neutral` branch and its accepted CLI label. The official
branch and evaluator computation are unchanged. A different diff aborts reuse.

Failure of any check forbids reuse and aborts before GPU work. It does not
silently regenerate official.

## Variant boundary and gates

The only intended edit-command differences are the alpha variant, shuffle salt,
and variant-specific output/audit paths. Every variant uses the official
empty-string negative. Diagnostics compare all per-concept raw MI, thresholds,
and pre-aggregate alpha tensors exactly across variants. They also check every
non-alpha command input, RNG call count and tensor signature, aggregate coverage,
constant means, shuffled multisets, all-one identity vectors, `trace(B)`,
`||B||_F`, and all tensors in the final checkpoint for finiteness.

`no_informax` is omitted. Setting only the final aggregate vector to zero would
leave the per-concept Informax-weighted UCE accumulation active, while disabling
Informax globally would change more than the requested final `B`. Neither is a
clean match to an established No-Informax ablation.

## Local safety

No Python or model workload is run on the Mac. `validate_static.sh` uses shell
parsing, source inspection, JSON parsing through Ruby, and hash checks only.
Actual editing, image generation, and evaluation are confined to the detached
GPU-server worker after the user activates Conda `MU`.
