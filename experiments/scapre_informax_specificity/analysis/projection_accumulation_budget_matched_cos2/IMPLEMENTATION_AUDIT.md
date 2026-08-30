# Budget-matched direct-cos2 implementation audit

## Qualification of the requested intervention

The production editor has exactly two relevant source expressions, one for
`to_v` and one for `to_k`:

```text
mat1_agg += erase_scale * (for_mat1 * row_w_c)
```

Both occur after the official accumulation Informax helper returns `row_w_c`.
The established experiment wrapper already replaces only these two expression
sites in an in-memory copy of the editor source. It does not edit the file on
disk. Aggregate Informax is earlier and separate, so final `row_w_max` is not
intercepted.

At each intercepted site, the wrapper receives the exact production tensors
`for_mat1`, `row_w_c`, `W_old`, `c_vec`, `empty_vec`, and `for_mat2`. Therefore
the requested budget match can be computed at the same stage without choosing
a different representation or weight tensor.

## Frozen treatment

```text
d = c_vec - empty_vec
r = ((W_old.float() @ d.float()) ** 2)
    / ((row_norm_sq + 1e-8) * (d_norm_sq + 1e-8))

C_official = for_mat1 * row_w_c
C_geo = for_mat1 * r.view(-1, 1).to(row_w_c dtype/device)
lambda = ||C_official||_F / (||C_geo||_F + 1e-8)
C_new = lambda * C_geo
```

The official branch returns `C_official`; the treatment branch returns
`C_new`. No clamp, fallback, score normalization, temperature, power, sweep,
retain concept, or concept-specific rule is applied.

## RNG and non-treatment isolation

Official Informax runs before every interception in both branches. The wrapper
retains the established legacy/isolated five-seed RNG handling, including the
same Gaussian draw signatures, ordering, returned tensors, and discarded
legacy draws. Budget arithmetic is deterministic and is executed in both the
official and treatment wrapper runs, so it does not alter RNG consumption.

The existing paired audit compares, bitwise where applicable:

- all production Informax diagnostics;
- target and empty embeddings;
- `W_old`, `for_mat1`, and `for_mat2` inputs;
- S/R/CCt/PiC and geometry inputs;
- global RNG states and entropy positions;
- aggregate `row_w_max` and pre-solve RNG state;
- normalized edit command, source substitution count, and production hash.

The new audit additionally compares the complete budget-match record between
official and treatment runs.

## Fail-closed qualification

For all 320 seed-20260820 concept/matrix records, qualification requires:

- finite, non-constant raw cos2;
- finite official, geometric, and matched contributions;
- both official and geometric contribution norms greater than `1e-8`;
- finite positive lambda;
- finite matched alpha and norm ratio;
- `torch.isclose(||C_new||_F, ||C_official||_F, rtol=1e-5, atol=1e-7)`;
- finite, non-no-op checkpoint;
- byte-unchanged production editor;
- complete RNG/input isolation and bitwise-identical final `row_w_max`.

The tolerance is an integrity tolerance only. It does not change `C_new` and is
not swept. Any failure stops before formal image generation.

Using the already completed direct-cos2 seed-20260820 norm diagnostics as a
read-only numerical preflight, all 320 geometric norms are well above epsilon
(minimum `3.4342`). The implied lambda range is `0.1815` to `36.4191`, and the
formula-implied norm-ratio range is `0.9999999971` to `0.99999999999`. These
values were not used to alter the frozen formula or tolerance.

## Protocol and lifecycle

The worker reuses the integrity-validated official five-seed scores and the
same 3,000 ordered Confuse5 rows per seed. Only 15,000 treatment images are
new. V1 and direct-cos2 aggregate metrics are hash-validated and reported as
descriptive context; official remains the sole formal baseline.

The detached worker automatically advances from passed qualification to the
five formal seeds, aggregates, writes integrity/validation artifacts, and
stops. Neither the launcher nor worker references or invokes a COCO runner.
