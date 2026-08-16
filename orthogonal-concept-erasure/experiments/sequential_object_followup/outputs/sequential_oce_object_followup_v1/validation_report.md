# Follow-up result validation

Overall assessment: **Needs revision / research use abandoned.**

The execution artifacts and calculations are internally consistent, but the
design does not identify the causal questions stated for the experiment. The
results are not ready for paper claims. See
[`../../../ABANDONED_SEQUENTIAL_OBJECT_EXPERIMENTS.md`](../../../ABANDONED_SEQUENTIAL_OBJECT_EXPERIMENTS.md).

## Scope and protocol checks

- Patch SHA-256 matched the supplied checksum: `5b522d974113829628c492a35ac0e6459a65b18a62778c0369a90e5822873ace`.
- Preflight planned exactly 45 formal cells and 7,000 new generations.
- Experiment 1 contains 10 direct-single cells (2,000 new images) and 20 sequential supplement cells (2,000 new images).
- Experiment 2 contains 15 clean-chain cells (3,000 new images).
- The official target-anchor mapping, fixed clean-chain order, prompt, sampler settings, and CIFAR-10 CLIP evaluator match the resolved protocol.
- The transfer does not include the launcher's `.run` exit-code file. Completion is instead supported by all four `run_state.json` phases, the terminal aggregation event, and the complete cell-level artifacts; this is an audit caveat, not a data discrepancy.

## Raw-data checks

- All 45 final cells contain exactly 200 prediction rows and seeds 42 through 241 exactly once.
- The 20 supplemented cells retain the original 100 rows unchanged and add exactly 100 rows at non-overlapping seeds 142 through 241.
- Metrics `correct` and `accuracy` were independently recomputed from every per-image CSV and match all 45 metrics files.
- Every row has ten class-probability columns whose sum is within numerical tolerance of 1.
- Combined raw files contain 2,000 direct-single rows, 4,000 sequential own-step rows, and 3,000 clean-chain rows.
- Experiment 1 and Experiment 2 tables were independently reconciled to the per-cell metrics.
- No generated image remains outside `figures/`; manifests and evaluator outputs remain auditable.

## Interpretation check

- Experiment 1 is not a matched prior-edit comparison. Direct-single omits the
  elephant retain request used by the sequential chains. Airplane already
  differs at W01, where no prior edits exist (0.040 sequential versus 0.015
  direct), demonstrating that prior edit history is not the only changed factor.
- Target identity, fixed order, and step number are confounded, so the observed
  cross-target pattern cannot identify an accumulation effect.
- Dog is worse in the original sequential own-step comparison, while bird is not.
- In the clean chain, dog does not resurge; bird does, rising from 0.475 to 0.735. Airplane also rises from 0.030 to a maximum of 0.180.
- The clean chain excludes later direct erasure of an anchor, but it is not a
  matched counterfactual to the original chain because order, step, retain
  history, and later edits also change. It cannot isolate why the original dog
  trajectory occurred.
- Bird and airplane provide sequence-specific descriptive evidence that later
  direct anchor erasure is not necessary for CIFAR-10 CLIP target accuracy to
  rise. They do not establish a universal persistence failure.

## Blocking disposition

- Do not use Experiment 1 differences as causal effects of previous edits.
- Do not use the elephant ceiling result to choose Retain Once over Retain
  Always; the required Retain Never control is absent.
- Do not generalize the fixed-chain trajectories beyond their tested orders.
- Preserve the raw artifacts for audit, but treat this experiment family as
  abandoned rather than adding more runs to it.

No artificial significance threshold was used; statements above report raw accuracies and absolute differences.
