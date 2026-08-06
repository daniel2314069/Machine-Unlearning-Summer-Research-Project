# OCE target–anchor correspondence diagnostic

The current canonical experiment evaluates the unchanged upstream/official
subspace objective. The paired vector-wise Eq. 6 implementation is retained
only as an ablation. It does not modify `oce.py`.

All Python commands use the repository-mandated Conda `py310` environment:

```bash
cd orthogonal-concept-erasure/experiments/correspondence_diagnostic

scripts/run_py310.sh -m pytest -q
scripts/run_py310.sh run_official_subspace.py preflight
scripts/run_py310.sh run_official_subspace.py tokenizer
scripts/run_py310.sh run_official_subspace.py feasibility
scripts/run_py310.sh run_official_subspace.py single
```

The official-subspace commands are resumable. The combined command below runs
only through the N=2 permutation check:

```bash
scripts/run_py310.sh run_official_subspace.py through-permutation \
  --n2-pairs cat_to_dog guitar_to_piano
```

It performs tokenizer audit, Original SD feasibility, five official
single-pair subspace screens, N=2 feature evaluation before image generation,
N=2 joint subspace/Eq. 6 image diagnostics, and the anchor permutation check.
It does not run the control set or N=5.

The old runner and its outputs remain available as the **preliminary
vector-wise screening (old concept set)**:

```bash
scripts/run_py310.sh run.py initial
```

Its vector-wise outputs are Eq. 6 ablations, not the official OCE baseline;
its old N=5 gate is not applicable to the new official-subspace experiment.

Canonical output:

`outputs/official_subspace/`

The output contains the exact pair/prompt/seed inputs, resolved parameters,
edited safetensors, saved layer rotations, per-image CSV, screening JSON/CSV,
seed-aligned grids, feature/image heatmaps, permutation diagnostics, artifact
validation, and the Markdown report.
