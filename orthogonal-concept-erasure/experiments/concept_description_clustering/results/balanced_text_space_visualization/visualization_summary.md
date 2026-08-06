# Balanced Text-Space Visualization Summary

## PCA explained variance

| Representation | PC1 | PC2 | PC1 + PC2 |
|---|---:|---:|---:|
| Unsuffixed EOT | 5.67% | 4.55% | 10.21% |
| Fixed suffix | 6.28% | 4.65% | 10.93% |

The EOT PC3 explains 3.99%. PC1–PC3 together explain 14.20%.

## Visible structure

The plots use the same 400 balanced descriptions and the existing spherical-k-means assignments. The EOT views show broad animal-associated structure with substantial local overlap; fox is the least isolated class and visibly intersects neighboring small-animal regions. The fixed-suffix views retain animal-associated structure but show broader mixing, consistent with their lower unchanged 768-dimensional ARI and matched accuracy.

Across the EOT t-SNE runs with seeds 0, 1, and 42, the broad class neighborhoods and recurring overlap patterns remain visible, while orientation, local arrangement, apparent gaps, and exact boundary shapes change. Seed 42 remains the predetermined main result; no seed was selected for visual quality.

## Metric and source audit

- Dataset: `/home/daniel1012/projects/machine_unlearning/orthogonal-concept-erasure/experiments/concept_description_clustering/balanced_paired/outputs/secondary_8x50/accepted_descriptions.jsonl`
- Dataset SHA-256: `6f459fa9e73f80163145813ecd8cd32c4216e1c4f77f6795925bf06676794a0c`
- Unsuffixed EOT: ARI 0.6338; matched accuracy 0.7750.
- Fixed suffix: ARI 0.4697; matched accuracy 0.7025.
- The existing raw cluster IDs and Hungarian mappings were reused. No clustering was refitted.
- PCA and t-SNE both received all 768 dimensions of the row-L2-normalized text-space vectors. No W0 projection or OCE operation was applied.

## Interpretation warning

PCA and t-SNE are low-dimensional visualizations only: the main PCA/t-SNE views use two dimensions, and the supplementary PCA view uses three. The clustering assignments and ARI/NMI were computed in the original 768-dimensional space. A clean-looking t-SNE does not prove that natural clusters exist, and t-SNE global distances, apparent cluster sizes, and empty gaps must not be interpreted literally.

## Presentation recommendation

Use `pca_eot_true_labels.png`, `tsne_eot_true_labels.png`, and `pca_eot_3d_true_labels.png` in the main presentation. Treat predicted-cluster views, fixed-suffix views, and the seed-robustness panel as comparison or appendix material.
