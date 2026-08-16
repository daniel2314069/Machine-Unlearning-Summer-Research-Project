# Matched sequential OCE pair-retain experiment

This is the replacement for the archived `sequential_object_persistence` and
`sequential_object_followup` studies. It answers only the matched two-edit
image-level questions in the preregistered five-pair schedule. It does not run
or report geometry diagnostics, commutators, Confuse5, joint erasure, kernels,
or semantic-overlap analyses.

## Fixed design

For every ordered pair `A -> B`, the runner evaluates all ten CIFAR-10 classes
at 200 images per class under:

1. Original SD v1.4 (generated once globally).
2. `W0 -> W_A`, using the unchanged repository object protocol.
3. `W_A -> W_A_then_B_baseline`, using the unchanged local retain list.
4. A second branch from the identical `W_A` checkpoint that appends `A` to the
   local retain list for erasing `B`.

The modified branch preserves `W_A c_A`: `oce.Orthogonal_Erase` reads the
currently loaded module weight as `W0` inside each call, and both Stage-2
branches explicitly load the same hashed Stage-1 checkpoint before editing.

The paper does not specify seeds. This experiment preregisters per-image seeds
42 through 241 and uses them for every class and every condition. Seeds 42 and
43 are also fixed before generation for qualitative retention. They are copied
from the formal images before safe cleanup, so no extra images are generated.

Formal count: 10 Original cells plus 10 orders x 3 edited conditions x 10
classes = 310 cells, or exactly 62,000 generated images.

## Server launch

After activating the GPU server project environment:

```bash
conda activate MU
cd orthogonal-concept-erasure/experiments/sequential_object_pair_retain
./run_pair_retain.sh --preflight
./run_pair_retain.sh
```

`--preflight` resolves and fingerprints the exact cached SD and CLIP snapshots,
checks `Cg.pt`, source hashes, package versions, paths, counts, and schedule. It
does not edit checkpoints or generate images. Review `run_manifest.json` before
the detached formal launch.

The launcher uses the active environment's `python`, starts a detached `nohup`
worker, and returns promptly. It is then safe to close the SSH terminal. It does
not protect against server reboot or administrator termination.

Status after reconnecting:

```bash
./run_pair_retain.sh --status
```

The runner is restart-safe at checkpoint and evaluator-cell boundaries. A
protocol/source/model-snapshot fingerprint prevents a changed run from reusing
or overwriting the output directory.

## Outputs

Results are written to:

```text
outputs/sequential_oce_pair_retain_v1/
├── per_class_results.csv
├── summary.csv
├── stage1_paper_metrics.csv
├── run_manifest.json
├── final_validation.json
├── events.jsonl
├── raw/all_predictions.csv
├── checkpoints/
└── qualitative/
```

Formal PNGs are deleted only after the 200 predictions, seed list, recomputed
accuracy, metrics, and qualitative-copy requirements for that cell validate.
Raw predictions, manifests, metrics, checkpoints, logs, fixed qualitative PNGs,
and contact sheets remain.

The final qualitative archive is written outside git at:

```text
/home/tslin/Documents/jupyter_data/anLi/tmp/sequential_oce_pair_retain_v1/qualitative_samples.tar.gz
```
