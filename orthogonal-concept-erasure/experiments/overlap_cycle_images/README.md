# OCE overlap/cycle image-only experiment

This directory contains a visual-only experiment using the repository's
unchanged official OCE subspace objective.

Run from the OCE repository with the required `py310` Conda environment:

```bash
conda run -n py310 python experiments/overlap_cycle_images/run_visual_experiment.py
```

The run intentionally executes every requested case even when a single-pair
mapping is visually weak. It creates images and seed-aligned grids only; it
does not compute CLIP, LPIPS, feature matrices, confusion matrices, heatmaps,
permutation checks, or vector-wise comparisons.
