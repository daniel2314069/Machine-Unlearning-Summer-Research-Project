# OCE solver-audit to AFR study archive

This directory is the single entry point for the completed Confuse5 research
branch that began with the OCE solver audit and ended with the AFR balls smoke.
The final proposed editor result is negative (`AFR-I0`), but the diagnostic
chain and ablations remain valid research artifacts.

No tracked result is moved from its original location: those paths are part of
the recorded provenance and are consumed by later audit code. This index groups
them logically without breaking hashes, imports, or absolute paths recorded on
the GPU server.

## Final status

See [`CONCLUSION.md`](CONCLUSION.md) for the scientific interpretation and stop
decision.

| Stage | Scope | Result | Report | Machine-readable data |
|---|---|---|---|---|
| Solver convention/rank audit | 3 groups x 16 layers | Outcome B | [`solver_audit/REPORT.md`](../solver_audit/REPORT.md) | [`results.csv`](../solver_audit/results.csv) |
| Exact orthogonal control | 3 groups x 16 layers | Outcome D2 | [`REPORT_exact_control.md`](../solver_audit/exact_orthogonal_control/REPORT_exact_control.md) | [`results_exact_control.csv`](../solver_audit/exact_orthogonal_control/results_exact_control.csv) |
| Anchor-minimum exact control | 3 groups x 16 layers | Outcome E1 | [`REPORT_anchor_min_control.md`](../solver_audit/exact_orthogonal_control/anchor_min_control/REPORT_anchor_min_control.md) | [`results_anchor_min_control.csv`](../solver_audit/exact_orthogonal_control/anchor_min_control/results_anchor_min_control.csv) |
| AFR C/F/G matrix QA | 3 groups x 16 layers | AFR-GO | [`AFR REPORT.md`](../afr/results/afr_balls_smoke_v1/REPORT.md) | [`matrix CSV`](../afr/results/afr_balls_smoke_v1/matrix/results_c_f_g.csv), [`gate`](../afr/results/afr_balls_smoke_v1/matrix/gate.json) |
| Conditional balls smoke | 1,800 images | **AFR-I0** | [`AFR REPORT.md`](../afr/results/afr_balls_smoke_v1/REPORT.md) | [`summary`](../afr/results/afr_balls_smoke_v1/balls_smoke/summary.json), [`LPIPS`](../afr/results/afr_balls_smoke_v1/balls_smoke/anchor_lpips.json), [`evaluations`](../afr/results/afr_balls_smoke_v1/balls_smoke/evaluations/) |

## Implementation

- [`solver_audit.py`](../solver_audit.py)
- [`exact_orthogonal_control/run.py`](../solver_audit/exact_orthogonal_control/run.py)
- [`anchor_min_control/run.py`](../solver_audit/exact_orthogonal_control/anchor_min_control/run.py)
- [`afr/core.py`](../afr/core.py)
- [`afr/runner.py`](../afr/runner.py)

Production `oce.py` was not changed by this branch.

## GPU-server artifacts

Run [`snapshot_server_artifacts.sh`](snapshot_server_artifacts.sh) on the GPU
server to create a non-destructive archive entry point. It copies the compact
tracked evidence, links the large runtime trees in place, and records file
sizes, SHA-256 values, Git state, and disk usage. It does not move or delete
images, checkpoints, K0, or logs. The payload links cover the qualified primary
runtime, AFR balls-smoke runtime, and the earlier invalidated pilot archive.

```bash
conda activate MU
cd orthogonal-concept-erasure
./experiments/confuse5_single_vs_joint/study_archive/snapshot_server_artifacts.sh
```

The default server snapshot is created at:

```text
experiments/confuse5_single_vs_joint/outputs/failed_oce_afr_study_archive_v1/
```

That runtime directory is intentionally ignored by Git. The durable compact
results are already tracked in the paths listed above.
