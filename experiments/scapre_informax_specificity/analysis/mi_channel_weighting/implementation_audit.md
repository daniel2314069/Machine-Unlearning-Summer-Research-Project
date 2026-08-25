# ScaPre Informax implementation audit

Audit date: 2026-08-25 (Asia/Taipei)

Repository HEAD: `c3cb73e8cfd530cad084058591426fabffa8e3ee`

Branch: `main` tracking `origin/main`

Working tree at audit start: clean

## 1. Paper formula

ScaPre v4, Sec. 4.2, Eq. (6) defines a binary empirical MI in natural-log
units. For multiple concepts it states

```text
MI_i = max_k MI_i^(k)
alpha_i = MI_i / max_j MI_j.
```

The paper defines `z = 1{a_i(s) > tau_i}` but does not specify in Eq. (6)-(7)
the repository's channel z-score, sigmoid temperature, or power transform.
Paper source: arXiv:2601.06162v4, Sec. 4.2, Eq. (6)-(7).

## 2. Repository flow

The full Confuse ScaPre command uses `scapre/edit/erase_scale.py`, not the
separate `erase.py` path (`scapre/script/erase.sh:71-86`). For each target,
projection, and layer, the effective path is:

```text
target text -> final content-token CLIP vector c: (d_in,)
empty text  -> CLIP vector at token index 1: (d_in,)
5 x (c + 0.01 N(0,I)) and 5 x (empty + 0.01 N(0,I)): (10,d_in)
W_old @ samples.T: (d_out,10)
lower median over dim=1: (d_out,1)
Z = (activation > median): (d_out,10)
smoothed empirical 2x2 MI in natural logs: (d_out,)
z-score over the d_out channels of this one concept/matrix call: (d_out,)
sigmoid(z / 0.7) ** 8: (d_out,)
max over the 10 concept masks: (d_out,1)
```

The helper standalone defaults are `noise_sigma=0.05` and `p=2`, but the
Confuse call sites pass the edit configuration: `noise_sigma=0.01`, `p=8`,
`num_pos=5`, and `T=0.7`.

## 3. Exact code locations

- Concept and empty vectors: `scapre/edit/erase_scale.py:403-417`.
- Official pseudo-samples and Gaussian RNG: lines `166-180`.
- Activation, lower-median threshold, and strict `>` state: lines `181-184`.
- Four pseudocount-smoothed cells and natural-log MI: lines `185-199`.
- Channel-wise mean/std z-score, temperature and power: lines `200-202`.
- Fixed production `5`, `0.7`, configured `p` and noise: lines `469-492`.
- `to_v` concept calls and post-soft-mask max: lines `540-557`.
- `to_k` identical control flow: lines `659-676`.
- A second, fresh stochastic call per concept is used for UCE accumulation:
  `to_v` lines `567-600`; `to_k` lines `685-722`.
- Diagnostic metadata: lines `750-763`.

`torch.std()` is called without a `dim` argument on the one-dimensional
`(d_out,)` MI vector, so it uses all output channels and PyTorch's default
sample-standard-deviation correction. There is no normalization across layers,
projections, samples, or concepts.

## 4. Dimensions and aggregation order

For each cross-attention matrix `W_old in R^(d_out x d_in)` and target `k`:

| Step | Tensor | Reduction dimension |
| --- | --- | --- |
| pseudo-samples | `(2n,d_in)` | none |
| activations | `(d_out,2n)` | matrix multiplication over `d_in` |
| threshold | `(d_out,1)` | median over sample dimension `dim=1` |
| binary state | `(d_out,2n)` | elementwise strict `>` |
| raw MI | `(d_out,)` | counts over the `2n` samples |
| z-score | `(d_out,)` | mean/std over all `d_out` channels |
| per-concept soft mask | `(d_out,1)` | elementwise sigmoid/power |
| final aggregate mask | `(d_out,1)` | max over concepts after soft masking |

`to_v` and `to_k` use the same helper and ordering, but their weight matrices
and random draws are distinct. Production also draws independent pseudo-samples
for the aggregate and accumulation stages; it does not reuse a single MI tensor.
The requested `249,600`-observation gate corresponds to aggregate-stage raw MI
only: 10 concepts times the sum of output channels over 16 layers and two
projections.

## 5. Paper/repository difference

The difference is real. The paper takes `max` over per-concept raw MI first and
then divides by the largest channel MI. The repository instead z-scores each
concept's MI across channels, applies `sigmoid(z/0.7)^p`, and only then takes a
concept-wise maximum. Paper-style and repository-style results must therefore
be computed and reported separately. This audit does not change either method.

History is consistent with this reading: the core MI helper traces to the
initial import commit `f6e5304`; commit `8c58b5b` added diagnostic returns and
the matched-negative experimental branch without changing the official MI
arithmetic. The later superclass work likewise leaves the official branch
unchanged.

The formal artifact stores raw MI and alpha after CUDA execution. Reapplying
the z-score to the stored raw MI on CPU is useful as a transparency diagnostic,
but is not a bitwise identity test: mean/std reductions may differ by backend,
and the subsequent eighth power can amplify small z-score differences. The
integrity gate therefore controls on the registered raw-MI counts and saved
alpha distribution, while recording the CPU recomputation difference.

## Audit decision

No major error was found in the prior understanding, so the integrity gate may
proceed. The analysis is explicitly restricted to the official empty-string
neutral and aggregate-stage MI; it does not execute model editing or diffusion
generation.
