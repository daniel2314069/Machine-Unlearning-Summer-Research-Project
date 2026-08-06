# EOT Local-Geometry Diagnostic

## 1. Motivation

This diagnostic tests whether the final balanced EOT text space contains strong local animal neighborhoods that a single-centroid clustering model may not capture. It has only two parts: post-hoc k-nearest-neighbor label purity and cosine kNN-graph spectral clustering. The earlier t-SNE view motivates the question but is not used as analytical input.

The result is mixed: local purity is high, and the primary spectral run exceeds the cached spherical baseline, but the spectral ARI drops below baseline when the graph expands to 15 neighbors.

## 2. Data and Representation

The analysis uses the frozen balanced dataset at `/home/daniel1012/projects/machine_unlearning/orthogonal-concept-erasure/experiments/concept_description_clustering/balanced_paired/outputs/secondary_8x50/accepted_descriptions.jsonl` (SHA-256 `6f459fa9e73f80163145813ecd8cd32c4216e1c4f77f6795925bf06676794a0c`): 400 descriptions, with 50 each for cat, dog, fox, bear, wolf, rabbit, deer, and horse. Every calculation uses the row-L2-normalized 400×768 unsuffixed-EOT matrix. No fixed suffix, W0 projection, OCE operation, PCA coordinate, or t-SNE coordinate is used.

The cached spherical baseline is ARI 0.6338, NMI 0.7047, and matched accuracy 0.7750. Its existing assignments were verified; spherical k-means was not refitted.

## 3. kNN Purity Diagnostic

kNN purity is not clustering. Cosine neighbors are found from the normalized 768D vectors without labels; labels are joined only afterward to calculate the same-animal fraction. The balanced random-label expectation among the other 399 samples is 49/399 = 0.1228.

| k | Mean purity | Median purity | SD | Random baseline |
|---|---|---|---|---|
| 5 | 0.8195 | 1.0000 | 0.2386 | 0.1228 |
| 10 | 0.7200 | 0.8000 | 0.2295 | 0.1228 |
| 20 | 0.6105 | 0.6500 | 0.2045 | 0.1228 |

At k=10, the observed overall mean purity is 0.7200, compared with 0.1228 under the balanced random-label reference. Individual neighbor identities and distances are retained in `knn_neighbors.csv`.

## 4. Cosine kNN Graph

Each graph connects a pair when either endpoint selects the other. Edge weight is non-negative cosine similarity, and the diagonal is zero. Labels are not available during neighbor selection or graph construction.

| Neighbors | Edges | Components | Component sizes | Min degree | Mean degree | Max degree |
|---|---|---|---|---|---|---|
| 5 | 1402 | 1 | [400] | 5 | 7.01 | 18 |
| 10 | 2746 | 1 | [400] | 10 | 13.73 | 36 |
| 15 | 4106 | 1 | [400] | 15 | 20.53 | 56 |

The primary graph uses exactly 10 cosine neighbors. Its connected-component structure is recorded rather than repaired or silently changed.

## 5. Spectral Clustering

Spectral clustering groups points using connectivity in the precomputed cosine kNN graph rather than one centroid per cluster in the original space. Every run fixes the number of clusters at eight and uses `assign_labels="kmeans"`; no label enters fitting.

| Method | Graph neighbors | Seed | ARI | NMI | Matched accuracy |
|---|---|---|---|---|---|
| Spherical k-means (cached) | — | 314159 | 0.6338 | 0.7047 | 0.7750 |
| Spectral clustering | 5 | 42 | 0.6803 | 0.7291 | 0.8250 |
| Spectral clustering | 10 | 0 | 0.6775 | 0.7332 | 0.8450 |
| Spectral clustering | 10 | 1 | 0.6832 | 0.7360 | 0.8475 |
| Spectral clustering | 10 | 42 | 0.6775 | 0.7332 | 0.8450 |
| Spectral clustering | 15 | 42 | 0.6200 | 0.7082 | 0.8200 |

The primary 10-neighbor, seed-42 run obtains ARI 0.6775, NMI 0.7332, and matched accuracy 0.8450.

## 6. Comparison with Spherical K-Means

The primary spectral run changes ARI by +0.0437 and matched accuracy by +0.0700 relative to the cached spherical baseline. All predetermined neighborhood and seed runs are shown above; no run was selected for being visually or numerically strongest.

At seed 42, ARI is 0.6803, 0.6775, and 0.6200 for 5, 10, and 15 graph neighbors. Across the three 10-neighbor seeds, ARI ranges from 0.6775 to 0.6832, and matched accuracy ranges from 0.8450 to 0.8475.

## 7. Per-Animal Results

| Animal | k=10 mean purity | k=10 median purity | Primary spectral recall |
|---|---|---|---|
| cat | 0.7520 | 0.8000 | 0.9000 |
| dog | 0.6820 | 0.7000 | 0.8400 |
| fox | 0.6760 | 0.7000 | 0.8200 |
| bear | 0.6540 | 0.7000 | 0.9000 |
| wolf | 0.7260 | 0.8000 | 0.8200 |
| rabbit | 0.7200 | 0.7000 | 0.9600 |
| deer | 0.7800 | 0.9000 | 0.5600 |
| horse | 0.7700 | 0.8000 | 0.9600 |

The table pairs local k=10 purity with recall from the predetermined primary spectral run. Purity describes local neighborhoods; recall describes the post-hoc Hungarian-matched global clustering and should not be treated as the same quantity.

## 8. Interpretation

**Case D.** Local purity is high, and the 10-neighbor result is stable across the three predetermined seeds, but robustness to graph neighborhood size is mixed. The 5- and 10-neighbor graphs improve ARI, whereas the 15-neighbor graph falls below the cached ARI baseline. The graph result is therefore neighborhood-size-sensitive and is not robust evidence against spherical k-means.

This result describes local and graph-based organization in the original normalized EOT space. It does not establish that the eight concepts are natural modes, and it does not validate separation merely because t-SNE appears clean.

## 9. Limitations

- The number of clusters is fixed to eight from the experimental design.
- kNN purity uses labels only as a post-hoc diagnostic and is not an unsupervised clustering score.
- Spectral clustering depends on graph construction and random k-means label assignment; the predetermined neighborhood and seed checks expose only limited sensitivity.
- The balanced-label baseline is an expectation, not an inferential significance test.
- No image behavior, W0 geometry, OCE behavior, or representation other than unsuffixed EOT is tested.
