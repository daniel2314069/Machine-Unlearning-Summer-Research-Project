# Follow-up result validation

Overall assessment: **Ready to share with the stated target-dependent caveat.**

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

- Experiment 1 is mixed: some later sequential requests are less effective, especially truck and ship, but automobile and bird improve and cat/deer are unchanged. A universal degradation claim is not supported.
- Dog is worse in the original sequential own-step comparison, while bird is not.
- In the clean chain, dog does not resurge; bird does, rising from 0.475 to 0.735. Airplane also rises from 0.030 to a maximum of 0.180.
- The clean chain therefore retains target-dependent previous-erasure persistence evidence, but does not reproduce the earlier dog case.
- Since bird and airplane rise even though their anchors are not later erase targets, later anchor erasure alone cannot explain all of the original observation. Other anchor interactions remain possible.

No artificial significance threshold was used; statements above report raw accuracies and absolute differences.
