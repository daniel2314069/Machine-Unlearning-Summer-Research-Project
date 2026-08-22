# ScaPre Informax Specificity Experiment

This directory implements a one-variable comparison of the public ScaPre
Informax behavior against matched similar-retain negatives.

- `official`: five empty-prompt negative pseudo-samples per target.
- `matched_retain`: five negative pseudo-samples whose base vectors are the
  three same-group retain concepts, assigned in fixed round-robin order
  (`2/2/1`).

Everything after the negative base-vector selection remains on the repository
code path: noise shape/count, median threshold, binarization, empirical MI,
z-score/sigmoid/power transformation, per-concept calls, max aggregation,
spectral regularization, solver, and Bures geometry step.

Read `AUDIT.md` before interpreting results.

## Formal result

The full five-group run completed successfully. Under the frozen decision rule,
the result is **SUPPORTED**: matched-retain increased aggregate Preserve
Accuracy by 1.11 percentage points, improved aggregate Unlearn Accuracy by 2.33
points (lower is better), and increased Overall Accuracy by 1.51 points. Four
groups improved preservation and the fifth tied.

The complete audit, exact metrics, all 25 concept results, limitations, and
mechanism diagnostics are in [`results/summary.md`](results/summary.md). The
tracked `results/` directory also contains the aggregate tables, top-channel
indices, Informax diagnostic table, and lightweight reproducibility manifests.
Raw per-image scores and raw MI/alpha tensors remain in the verified formal
archive rather than Git.

The fixed five-seed follow-up is implemented under
[`seed_robustness/`](seed_robustness/README.md). It reuses the validated
`20260820` scores and generates only seeds `20260821`–`20260824` while holding
generation keys and all non-Informax randomness fixed.

## Protocol status

The official public ScaPre repository does not contain the complete 25-class
Confuse5 prompt/seed asset. Its `imagenet-15.csv` contains only 15 classes, and
its public evaluator defaults to 130 images even though paper Table 7 is
consistent with 120 images per concept.

The formal configuration is therefore explicitly labeled a **project-specific
reconstruction**, not an exact reproduction of paper Table 7:

- all 25 paper concepts and the paper target/retain assignment are used;
- 120 images per concept are used;
- the established project asset
  `orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/datasets/imagenet-confuse5-derived-25.csv`
  is hash-pinned and supplies all rows;
- its original 15 classes are byte-derived from the public ScaPre CSV;
- for each missing retain class, the asset reuses the ordered seeds of the
  same-group retain class that is present in the public CSV; the formal run uses
  the first 120 rows per concept;
- both variants always use the identical generated protocol CSV.

This reconstruction is appropriate for the paired falsification comparison,
but its absolute accuracies must not be represented as an exact Table 7
reproduction.

The 25-class asset was created in this project at commit `06ae690` by
`build_derived_dataset.sh`; it is not an author-released ScaPre supplement.

## Fresh-clone GPU-server workflow

The only machine-level prerequisite is the existing Conda `MU` environment.
Scripts never activate Conda or hard-code its Python path. An optional
model-free local check (safe on the Mac and never invoking Python) is:

```bash
experiments/scapre_informax_specificity/validate_local.sh
```

It requires `jq` and `rg`. The
server setup performs its own Python/stdlib config and protocol preflight, so
this optional Mac check is not a server prerequisite.

From the repository root on the GPU server:

```bash
conda activate MU

experiments/scapre_informax_specificity/setup_server.sh
```

`setup_server.sh` performs all remaining setup:

- verifies that `MU` is active and uses `command -v python`;
- installs pinned CUDA PyTorch and the experiment's minimal dependencies;
- verifies CUDA and package consistency;
- resolves the Stable Diffusion v1.5 revision and downloads only the
  componentized Safetensors files used by the pipeline (about 5.5 GB; root
  checkpoints and duplicate bin/fp16/non-EMA/ONNX/Flax artifacts are excluded);
- downloads and hash-checks Torchvision's default ResNet-50 weights;
- records resolved model/package provenance plus every downloaded model file
  and its size, and rejects required assets above a 7 GiB safety limit;
- validates construction of the full 25-class protocol.

If Hugging Face requires authentication, export `HF_TOKEN` in the shell before
running setup. Do not put a token into any script or config file.

### Required smoke test

The smoke profile edits only the dog group's two targets and evaluates two
images for each of its five concepts. It verifies the complete code path but is
not a scientific result.

```bash
experiments/scapre_informax_specificity/run_server.sh smoke
experiments/scapre_informax_specificity/status_server.sh
```

After the launcher's health check succeeds, terminal and SSH sessions may be
closed safely. To resume a failed run directory after resolving an external
problem, use its original ID:

```bash
experiments/scapre_informax_specificity/run_server.sh smoke RUN_ID --resume
```

The status output prints the exact run directory and recent log. Confirm that
the smoke run is `completed`, exit code is `0`, and its summary says no
scientific judgment before launching formal evaluation.

### Formal five-group experiment

```bash
experiments/scapre_informax_specificity/run_server.sh formal
experiments/scapre_informax_specificity/status_server.sh
```

The formal worker performs, in order:

1. protocol/config/source fingerprinting;
2. official and matched-retain edits from the same resolved base snapshot and
   edit seed;
3. normalized command comparison proving no other edit flag changed;
4. 25-class ResNet-50 generation/evaluation using identical prompts and seeds;
5. paired data validation;
6. per-concept, per-group, and aggregate official metrics;
7. raw MI/alpha diagnostics and top-channel overlap/correlation;
8. `results/summary.md` with one pre-registered formal judgment.

Evaluation resumes at row granularity. Existing images/results are reused only
when their checkpoint/protocol/generation fingerprint matches.

## Result packaging

After status reports `completed` with exit code `0`:

```bash
experiments/scapre_informax_specificity/package_results.sh \
  /absolute/path/printed/by/status
```

The archive is written only to:

```text
/home/tslin/Documents/jupyter_data/anLi/tmp
```

It includes tables, summary, configs, protocol, manifests, logs, scores,
Informax diagnostics, and a lightweight snapshot of every experiment source
file plus the modified core editor. It intentionally excludes generated
images, model checkpoints, caches, and downloaded weights. Original server
outputs are not deleted or moved.

The packaging script prints the archive's exact absolute path and SHA-256.

## Download to the Mac

Run this only on the Mac, substituting the actual SSH host and the exact archive
path/checksum printed by `package_results.sh`:

```bash
experiments/scapre_informax_specificity/download_results.sh \
  'tslin@ACTUAL_SERVER_HOST' \
  '/home/tslin/Documents/jupyter_data/anLi/tmp/ACTUAL_ARCHIVE.tar.gz' \
  'ACTUAL_64_CHARACTER_SHA256'
```

This is equivalent to the following exact command shape:

```bash
scp 'tslin@ACTUAL_SERVER_HOST:/home/tslin/Documents/jupyter_data/anLi/tmp/ACTUAL_ARCHIVE.tar.gz' \
  "$HOME/Downloads/ACTUAL_ARCHIVE.tar.gz"
```

The download script refuses to overwrite an existing local archive and verifies
the server-provided SHA-256 before reporting the full local path. It does not
delete the server archive or original outputs.

## Output layout

Each run is isolated under `runs/<profile>_<run-id>/`:

```text
actual_config.json
protocol.csv
protocol_manifest.json
run_manifest.json
controlled_ablation_check.json
provenance/                  # exact lightweight source snapshot
checkpoints/                 # not packaged
diagnostics/
stages/                      # edit hashes and exact commands
evaluation/
  official/{images,scores.csv,evaluation_manifest.json}
  matched_retain/{images,scores.csv,evaluation_manifest.json}
results/
  aggregate.csv
  per_group.csv
  per_concept.csv
  informax_diagnostics.csv
  top_channels.json
  result_manifest.json
  summary.md
server.log
exit_code
COMPLETED or FAILED
```

## Pre-registered interpretation

The decision rule is frozen in `config.json` before results exist:

- minimum aggregate Preserve gain: 1.0 percentage point;
- positive Preserve delta in at least four of five groups;
- Unlearn degradation non-inferiority margin: at most 2.0 percentage points.

A formal run emits exactly one of `SUPPORTED`, `NOT SUPPORTED`, or
`TRADE-OFF ONLY`. The internal Informax diagnostics never determine success.

## Final superclass-neutral ablation

The final experiment in this research line is contained under
`superclass_neutral/`. It reuses the completed five-seed official and
matched-retain score rows, generates only the new superclass-neutral variant,
and retains a small predeclared three-variant qualitative comparison set. Its
server lifecycle, packaging, image-retention policy, and exact mapping are
documented in `superclass_neutral/README.md` and `superclass_neutral/AUDIT.md`.
