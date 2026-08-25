# ScaPre Informax MI and channel-weighting diagnostic

Formal server run: `formal_20260825T092351Z`<br>
Model snapshot: Stable Diffusion v1.5 revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`<br>
Informax seeds: `20260820`–`20260824`

## 1. Implementation audit

The repository computes raw MI separately for every concept, projection, and
layer; z-scores that vector across its output-channel dimension; applies
`sigmoid(z/0.7)^8`; and finally takes a channel-wise maximum over concepts.
Paper Eq. (7) instead takes the concept-wise maximum of raw MI and then divides
by the maximum channel MI. The difference is real and the two paths below are
kept separate. Exact code locations, tensor dimensions, RNG behavior, and K/V
parity are in [implementation_audit.md](implementation_audit.md).

No production ScaPre source was modified. Its SHA-256 was identical before and
after the server analysis.

## 2. n=5 integrity check

The gate **passed** and exactly reproduced the registered aggregate-stage
diagnostic:

| Quantity | Reproduced value |
| --- | ---: |
| channel observations | 249,600 |
| MI approximately 0.0201 | 1,601 |
| MI approximately 0.0863 | 1 |
| MI approximately 0.1927 | 2,096 |
| MI approximately ln(2) | 245,902 |
| fraction numerically at ln(2) | 98.518431% |

Saved repository-alpha statistics were mean `0.00743361`, standard deviation
`0.00122901`, median `0.00747851`, p90 `0.00857593`, p95 `0.00870015`, p99
`0.00919989`, minimum `0`, and maximum `0.01101426`.

The unique MI≈0.0863 observation is `to_v`, layer 8, target `golden retriever`,
channel 1009, raw MI `0.0863045827`, saved threshold `-0.351055086`, and saved
alpha `2.1875e-36`. The legacy artifact stores neither its ten activations nor
its binary states. Its 2×2 table and whether ties or precision caused the value
therefore cannot be reconstructed defensibly; no cause is asserted.

The raw-MI gate and saved-alpha distribution match. Recomputing alpha from
CUDA-produced raw MI on CPU has a maximum elementwise difference of `0.003826`;
this is retained as a backend reduction-order diagnostic and was not used as a
bitwise gate.

## 3. Sample-size result

Each seed used one independently seeded 50-positive/50-neutral pool. Smaller n
used exact prefixes of that pool, so comparisons are nested and do not depend
on differently shaped random draws.

| n+n | observations | mean MI | fraction at ln(2) | repo alpha mean | median | p99 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5+5 | 249,600 | 0.684657 | 98.5094% | 0.007434 | 0.007578 | 0.009455 | 0.010789 |
| 10+10 | 249,600 | 0.683318 | 98.0369% | 0.007984 | 0.008172 | 0.010383 | 0.011259 |
| 20+20 | 249,600 | 0.682585 | 97.6294% | 0.008302 | 0.008504 | 0.010843 | 0.011420 |
| 50+50 | 249,600 | 0.682101 | 97.1658% | 0.008509 | 0.008776 | 0.011115 | 0.011465 |

The maximum-MI fraction falls only `1.3436` percentage points from n=5 to
n=50. This is **Case B**: finite n contributes slightly, but even 50+50 is
almost completely saturated. The dominant observation is that the official
target-versus-empty-neutral comparison is intrinsically easy to separate.

In the no-tie case, direct count-table enumeration gives exactly
`floor(n/2)+1` numerical MI values: 3, 6, 11, and 26 for n=5, 10, 20, and 50.
The server's original float32 enumeration listed symmetric values separately
when they differed only in their final arithmetic bits; the checked-in copies
coalesce those values at `1e-7`, without changing any MI observation. Exact
threshold ties occurred in 3.47%, 4.68%, 5.80%, and 7.26% of channel records,
respectively, which explains why the observed support can extend beyond the
no-tie enumeration.

Repo alpha moves upward and its upper quantiles separate modestly as n grows,
but this should not be mistaken for broad raw-MI discrimination: 97.17% of raw
MI remains at ln(2) at n=50.

## 4. Stability of n=5 maximum-MI channels

Across seeds, there were about 245,860–245,898 n=5 maximum-MI channels per
seed. Of those same channel identities:

| Larger nested pool | Still at ln(2) |
| ---: | ---: |
| n=10 | 99.5204% |
| n=20 | 99.1067% |
| n=50 | 98.6361% |

Thus the overwhelming majority of n=5 maxima are stable even after observing
ten times as many samples. Projection/layer/target denominators are in
[`max_mi_stability.csv`](results/max_mi_stability.csv).

## 5. Descriptive activation separation

For all 1,229,397 n=5 max-MI channel records, the closest cross-group raw
activation distance was:

| min | p01 | p05 | median | p95 | p99 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0000267 | 0.01477 | 0.07422 | 0.90332 | 3.4990 | 5.2539 | 15.2148 |

For example, the minimum is seed 20260822, `to_v`, layer 11, `tabby`, channel
339; the maximum is seed 20260821, `to_k`, layer 13, `lifeboat`, channel 247.
The large range confirms that equal saturated MI can coexist with very
different raw separations. Raw distance is channel-scale dependent and is only
a descriptive diagnostic—not a relevance score. Ten smallest and largest
examples are in
[`activation_distance_diagnostic.json`](results/activation_distance_diagnostic.json).

## 6. Paper-style max over concepts

All 1,023 non-empty subsets were enumerated for every seed. Across subsets of a
given size:

| m | subsets × seeds | fraction max MI | fraction paper alpha=1 | mean paper alpha |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 50 | 98.5094% | 98.5094% | 0.987751 |
| 2 | 225 | 99.9739% | 99.9739% | 0.999802 |
| 3 | 600 | 99.9997% | 99.9997% | 0.999998 |
| 4 | 1,050 | 100% | 100% | 1.000000 |
| 10 | 5 | 100% | 100% | 1.000000 |

Under the paper formula, max over concepts makes saturation essentially
complete by m=2 and exactly complete for every observed subset from m=4 onward.
This is not the repository's executed weighting path.

## 7. Repository-style max over concepts

The repository order—per-concept z-score, sigmoid/power, then max—produces a
smaller but monotonic upward shift:

| m | mean alpha | median | p95 | p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.007434 | 0.007562 | 0.008732 | 0.009632 |
| 2 | 0.007929 | 0.007928 | 0.008995 | 0.010002 |
| 3 | 0.008131 | 0.008102 | 0.009180 | 0.010218 |
| 5 | 0.008360 | 0.008308 | 0.009405 | 0.010482 |
| 10 | 0.008642 | 0.008541 | 0.009685 | 0.010789 |

More concepts therefore raise final repo weights because every channel has more
per-concept masks from which to take a maximum. The effect is visible but does
not force repo alpha to one, unlike paper normalization.

## 8. Official 50-concept configuration

The repository contains an unambiguous ordered ImageNet-Diversi50 configuration
in `scapre/script/erase.sh`; cumulative prefixes m=1,5,10,20,30,40,50 were
analyzed with its configured power 5.

Paper-style maximum MI rises from `98.5809%` at m=1 to `100%` by m=5 and stays
there. Repo-style mean alpha rises from `0.046020` at m=1 to `0.052913` at m=50;
median rises from `0.046233` to `0.052556`, and p99 from `0.055954` to
`0.061832`. These alpha magnitudes must not be compared directly with the
10-target power-8 run because the official 50-concept configuration uses power
5.

## 9. Limitations

- The new RNG streams are deterministic analysis streams, not replays of the
  production global RNG positions.
- Results concern Stable Diffusion v1.5, the selected concept lists, and the
  official empty-string neutral only.
- The 249,600-observation definition covers aggregate-stage MI; production's
  independently redrawn accumulation stage is outside this diagnostic.
- The legacy artifact prevents a defensible reconstruction of the sole
  MI≈0.0863 contingency table.
- Float16 activation ties and float32 MI arithmetic are faithfully retained;
  only the presentation count for numerically symmetric enumerated values was
  tolerance-coalesced after the run.

## 10. Next research question

Without changing ScaPre, the next useful diagnostic is whether the same
saturation and concept-max behavior persists across model checkpoints, concept
families, and repeated official-neutral pseudo-sample pools. That would test the
generality of the present finding while holding the MI estimator, threshold,
neutral construction, and channel transformation fixed.

## Result storage

Small and medium CSVs are checked in under [`results/`](results/). The complete
252 MB activation table and 42 MB repo-subset table are checked in losslessly as
`max_mi_activation_summary.csv.gz` and
`concept_count_repo_formula.csv.gz`; decompression restores the requested CSV
filenames byte-for-byte. The original completed server archive remains at
`~/Downloads/scapre_informax_mi_channel_weighting_formal_20260825T092351Z_20260825T113157Z.tar.gz`
with locally computed SHA-256
`7af397d0a2616307009df4735e852117dafc661b9ea55a038ba05f88cfbff150`.
