# Sequential OCE object follow-up summary

> **RESEARCH STATUS: ABANDONED / NOT FOR CLAIMS.** The 7,000 generations and
> evaluator artifacts passed their execution audits, but the comparisons do not
> identify the intended causal effects. See
> [`../../../ABANDONED_SEQUENTIAL_OBJECT_EXPERIMENTS.md`](../../../ABANDONED_SEQUENTIAL_OBJECT_EXPERIMENTS.md).

- New formal generations: **7,000** exactly.
- Every final formal cell: **200 unique predictions**.
- Evaluator: unchanged CIFAR-10 10-class CLIP ViT-B/32 protocol.
- No artificial significance threshold is applied; all comparisons below are raw accuracies and absolute differences.

## Experiment 1 — direct single from W0 vs sequential own-step

- `airplane`: direct=0.015, Retain Once own-step=0.040 (difference +0.025), Retain Always own-step=0.040 (difference +0.025)
- `automobile`: direct=0.900, Retain Once own-step=0.750 (difference -0.150), Retain Always own-step=0.695 (difference -0.205)
- `bird`: direct=0.570, Retain Once own-step=0.465 (difference -0.105), Retain Always own-step=0.400 (difference -0.170)
- `cat`: direct=0.000, Retain Once own-step=0.000 (difference +0.000), Retain Always own-step=0.000 (difference +0.000)
- `deer`: direct=0.000, Retain Once own-step=0.000 (difference +0.000), Retain Always own-step=0.000 (difference +0.000)
- `dog`: direct=0.005, Retain Once own-step=0.100 (difference +0.095), Retain Always own-step=0.080 (difference +0.075)
- `frog`: direct=0.050, Retain Once own-step=0.080 (difference +0.030), Retain Always own-step=0.060 (difference +0.010)
- `horse`: direct=0.075, Retain Once own-step=0.075 (difference +0.000), Retain Always own-step=0.365 (difference +0.290)
- `ship`: direct=0.055, Retain Once own-step=0.185 (difference +0.130), Retain Always own-step=0.180 (difference +0.125)
- `truck`: direct=0.075, Retain Once own-step=0.405 (difference +0.330), Retain Always own-step=0.430 (difference +0.355)

The largest positive raw differences were **truck** (+0.330 / +0.355),
**ship** (+0.130 / +0.125), and **dog** (+0.095 / +0.075). Horse had a large
positive difference only under Retain Always (+0.290). These must not be called
sequential penalties: direct-single omits the elephant retain request present in
the sequential chains, so the conditions differ in more than prior edit history.

Dog was therefore worse at its original sequential own step than direct-single. Bird was not: its two sequential own-step accuracies (0.465 / 0.400) were both below direct-single (0.570).

The direction was not consistent across targets. Automobile and bird were lower
under both sequential conditions, while cat and deer were unchanged. Because
the comparator is unmatched and each target occupies one fixed step, Experiment
1 cannot establish that previous edits caused later OCE requests to become less
effective.

## Experiment 2 — clean five-step persistence

Dog did **not** reproduce its earlier resurgence: W1=0.005 and its later values were 0.000, 0.000, 0.005, and 0.000; no later checkpoint exceeded W1.

Bird **did** reproduce an upward trajectory: W2=0.475, W3=0.660, W4=0.690, and W5=0.735 (raw increase +0.260).

Airplane also rose after its own erase step: W3=0.030, W4=0.180, and W5=0.145 (maximum raw increase +0.150).

Because none of these targets' anchors is erased later in this clean chain,
later erasure of an anchor is not necessary for a target-accuracy increase to
appear in at least this bird/airplane sequence. The clean chain is not a matched
counterfactual to the original chain: it also changes order, step, retain
history, and subsequent edit identities. It therefore does not isolate the
cause of the original dog/bird trajectories.

Compared with direct-single, clean-chain own-step accuracy was equal for dog (0.005 vs 0.005) and deer (0.000 vs 0.000), lower for bird (0.475 vs 0.570) and automobile (0.735 vs 0.900), and only 0.015 higher for airplane (0.030 vs 0.015). This chain does not show a consistent accumulation-driven loss of own-step effectiveness.

Overall, the clean chain records target- and order-specific CIFAR-10 CLIP
accuracy increases for bird and airplane, but not dog. Together with the
unmatched comparisons and fixed-order confounding, this experiment family does
not establish a general previous-erasure persistence problem.

The persistence table and per-target summary report every 200-image checkpoint accuracy. Conclusions are limited to this fixed repeat.

## Artifacts

- `inputs/cell_manifest.csv` and `inputs/planned_generation.json`
- `raw/*per_image_predictions*.csv` and per-cell evaluator outputs
- `tables/experiment1_comparison.csv`
- `tables/experiment2_persistence.csv`
- `tables/experiment2_per_target_summary.csv`
- `figures/experiment2_previous_erasure_heatmap.*`
- `figures/experiment2_trajectories.*`

Generated PNGs are removed only after the corresponding 200-row evaluator artifacts pass the final seed/count audit when delete-after-eval is active.
