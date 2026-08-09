# Confuse5 Single-vs-Joint OCE — official-repo primary rerun

This directory implements the gated rerun that asks one question: after a
Single OCE edit demonstrably erases its target, does putting both similar
targets from the same Confuse5 group into one Joint subspace cause additional
target residual or collateral damage to the three designated similar
non-targets?

The archived default-config run is not primary evidence. Its complete outputs
are preserved under
`archives/invalid_for_primary__pilot_default_config/`; the machine-readable
invalidation and the separately conditional Original reference are the only
archive files intended for version control.

## Provenance categories

The resolved config keeps three sources separate:

- **Official released repository behavior:** `lamb=10`, the released
  post-`UV^T` last-column determinant correction, object prompt expansion, and
  the official expansion ordering (all bare concepts, then per-concept extra
  templates).
- **Final-paper specification:** Appendix C object values
  `lambda_e=1000`, `lambda_0=50`, `lambda_r=1`, COCO token second moment, and
  heuristic semantic anchors.
- **Benchmark-specific Confuse5 choices:** the fixed `anchors.json`, the three
  similar non-targets as local `C_n`, matched 12,500 rows, four-target gate,
  and conditional reuse of legacy Original exact top-1.

Parser defaults are forbidden. The known paper/repo discrepancies are retained
in `config.json` and every checkpoint records a checkpoint-level primary
versus paper-literal diagnostic. The primary is `lamb=10` with correction on;
the diagnostic is `lamb=0` with correction off. There is no second 37,500-image
paper-literal run.

## Fixed anchors

| Target | Anchor |
|---|---|
| golden retriever | cocker spaniel |
| labrador retriever | beagle |
| tabby | lynx |
| tiger cat | lion |
| orange | banana |
| lemon | pineapple |
| yawl | canoe |
| lifeboat | ferry |
| soccer ball | basketball |
| volleyball | baseball |

Single and Joint checkpoints consume the same per-target mapping. Anchors are
not included in local retain concepts and cannot be tuned from edit outcomes.

## Strict stage order and stops

`pipeline.py all --skip-existing` performs:

```text
clean K0 -> anchor sanity -> 10 Single + 5 Joint checkpoints
         -> Original reproduction canary -> four-target smoke
         -> hard gate -> 37,500 edited images -> aggregate
```

The pipeline stops, writes a machine-readable gate, and never enters the full
stage in exactly these cases:

1. an anchor produces at least 4/8 exact target labels;
2. any newly rendered Original smoke PNG differs from its archived SHA-256;
3. any of labrador retriever, tabby, yawl, or volleyball has
   `Original correct - Single correct < 4` among its ordered first 32 rows.

Joint is measured but is not a smoke gate. Anchor, canary, and smoke PNGs are
retained. Formal edited PNGs are removed only after their complete per-image
metrics and hashes are durably saved.

The formal stage generates no Original images. Once the 128-image canary
passes, it loads the archived 25 × 500 exact top-1 shards. Legacy Original
target probability, raw logit, and top-5 remain explicitly unavailable.

## GPU-server execution

Do not run model stages on the local Mac. On the GPU server:

```bash
conda activate MU
cd orthogonal-concept-erasure

# Standard-library-only resolved plan (safe before launching the GPU job).
python experiments/confuse5_single_vs_joint/pipeline.py plan

# Recommended detached, strictly ordered run.
./experiments/confuse5_single_vs_joint/launch_detached.sh all --skip-existing

# Inspect the detached worker and gate/stage state.
./experiments/confuse5_single_vs_joint/status.sh
```

The launcher uses the active environment's `python`; it does not hard-code a
Conda environment. Subprocess stages inherit that same `sys.executable`.

Individual recovery/inspection stages are also available:

```bash
python experiments/confuse5_single_vs_joint/pipeline.py k0 --skip-existing
python experiments/confuse5_single_vs_joint/pipeline.py anchor-sanity --skip-existing
python experiments/confuse5_single_vs_joint/pipeline.py checkpoints --skip-existing
python experiments/confuse5_single_vs_joint/pipeline.py smoke --skip-existing
python experiments/confuse5_single_vs_joint/pipeline.py formal --skip-existing
python experiments/confuse5_single_vs_joint/pipeline.py aggregate
python experiments/confuse5_single_vs_joint/pipeline.py status
```

Failed K0/checkpoint artifacts are never resumed or overwritten; use a fresh
namespace after investigating a failure. Completed formal job shards can be
skipped by matching fingerprints.

## Output contract

All new artifacts live below `outputs/official_repo_primary_v1/`:

```text
resolved_pipeline_plan.json
artifacts/K0.pt
artifacts/K0.metadata.json
anchor_sanity/{gate.json,per_image.json,images/...}
checkpoints/<group>/{single/<target>,joint}/{weights.safetensors,metadata.json}
checkpoint_summary.json
original_canary/gate.json
smoke/{gate.json,per_image.json,images/...}
formal/resolved_plan.json
formal/manifests/*.json
formal/evaluations/{shards/*.json,per_class.csv,per_image.csv}
formal/aggregates/{summary.json,target_residual.csv,
                   similar_non_target_preservation.csv,
                   sibling_target_secondary.csv}
```

Each edited image shard retains exact top-1, target probability, raw target
logit, and top-5 labels/probabilities/logits. Primary reporting uses ResNet
exact top-1. `Joint - Single` target residual above zero means worse Joint
erasure; `Joint - Single` preservation below zero means additional Joint
collateral damage. The sibling target is secondary only.

Any later MSCOCO general-preservation run remains outside this pipeline and
must use the repository-level evaluation-reference registry before reusing an
Original CLIP/FID baseline.
