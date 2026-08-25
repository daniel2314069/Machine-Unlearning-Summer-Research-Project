# Original-W0 Geometry of Balanced Name-Free Animal Descriptions

## 1. Research Questions

This experiment asks whether each unchanged pretrained SD 1.4 cross-attention projection preserves or improves clustering of the final balanced 8×50 name-free dataset; whether explicit names are near their own description centroids under matched readouts; and whether the exact OCE name vector is near EOT description centroids under an intentionally cross-readout comparison.

## 2. Representations and the Readout-Mismatch Problem

The EOT description readout is the final EOT hidden state of the original unsuffixed description, selected at `attention_mask.sum(dim=1) - 1`. The fixed-suffix readout appends exactly ` This sentence describes the concept` and extracts the contextual hidden state of the shared final content word `concept`. The OCE name readout reproduces `oce.py`: it calls `pipe.encode_prompt` on the bare name and selects the final content token immediately before EOT at `attention_mask.sum() - 2`.

`OCE-last-token -> EOT-description centroid` intentionally compares different readout rules. It is operationally relevant to OCE, but it is not a matched semantic-distance measurement and its absolute values should not be compared with matched EOT without this qualification.

## 3. Original W0 Layer Inventory

Repository inspection found that `Orthogonal_Erase` edits only the 16 original `UNet` cross-attention `attn2.to_v` matrices. No `to_k` matrix is selected, so the main analysis contains `to_v` only. Layer shapes are 5×(320×768), 5×(640×768), 6×(1280×768); every input dimension is 768. Full ordered module names and per-layer before/after hashes are in `layer_inventory.csv` and `w0_immutability.json`.

## 4. Layer-wise Description Clustering

| Representation | Best W0 Layer | Text-space ARI | Best-layer ARI | Best-layer Accuracy |
|---|---|---|---|---|
| Unsuffixed EOT | L8 | 0.6338 | 0.6795 | 0.8175 |
| Fixed suffix | L9 | 0.4697 | 0.6049 | 0.7475 |

The best-layer choices are post-hoc maxima over all 16 layers. Relative to text space, the best EOT layer changes ARI by +0.0457; the best fixed-suffix layer changes ARI by +0.1351. All layers and both readouts are retained in `layer_clustering_metrics.csv`; clustering always used the original 400×layer-dimension vectors, never PCA.

The plot compares every layer with its representation-specific text-space baseline. The companion accuracy plot is `plots/layer_accuracy.png`.

## 5. Matched Prototype-to-Cluster Results

At the post-hoc best EOT-ARI W0 layer L8:

| Condition | Mean Own Rank | Rank-1 Names / 8 | Mean Margin |
|---|---|---|---|
| Matched EOT | 1.250 | 6 / 8 | 0.0886 |
| Matched fixed suffix | 1.750 | 6 / 8 | 0.0120 |
| OCE-last-token -> EOT-description centroid | 1.375 | 6 / 8 | 0.1081 |

Positive margin means the name is closer to its own true-label description centroid than to every other centroid. The typicality percentiles and all name-to-centroid distances are retained in the detailed CSVs.

The heatmap shows text space followed by every W0 layer, making stability across layers visible rather than reporting only the best layer.

## 6. OCE-Faithful Prototype-to-EOT Results

The OCE-faithful condition maps the exact bare-name last-content-token vector through each unchanged `W0` and compares it with the corresponding EOT-description centroids. This is labeled `OCE-last-token -> EOT-description centroid` throughout the outputs and is kept separate from both matched conditions.

Ranks, margins, nearest concepts, and within-class typicality percentiles for all names and layers are in `prototype_summary.csv`.

## 7. Readout-Mismatch Comparison

At L8, concepts whose rank-1 status changes between matched EOT and the OCE cross-readout are: none.

For fox, bear, the canonical name is not close to its name-free description centroid under either tested representation.

Across all 16 W0 layers, rank-1 status differs for 27/128 layer–concept pairs spanning 10/16 layers: 15 succeed only under matched EOT and 12 succeed only under the OCE cross-readout. The all-layer, per-concept comparison of both ranks and margins is saved in `readout_mismatch_summary.csv`. Cross-readout rows are never pooled with matched-readout rows.

## 8. Main Findings

- W0 effects are layer-specific: L8 improves EOT ARI by +0.0457, and L9 improves fixed-suffix ARI by +0.1351, while several other layers substantially degrade both clusterings. These post-hoc maxima do not imply that W0 created the concept structure.
- At the best EOT layer, matched EOT and OCE cross-readout have the same rank-1 status for all eight names, but the 27/128 all-layer disagreement count shows that readout choice is not generally invariant across W0 spaces.
- Every numerical output retains all layers. Selected-layer figures are presentation summaries, not a pre-registered layer choice.
- This experiment analyzes unchanged original matrices only. It does not load an edited checkpoint, compute `P W0`, generate images, or modify model parameters.

## 9. Limitations

The centroids use true labels post hoc and therefore characterize known classes rather than discovering them. Best-layer reporting is post hoc. Cosine distance depends on both the readout and the layer-specific projection, and vectors from different layers are never directly compared. Prototype proximity does not prove semantic identity, central representation, absence of information, or whether OCE can erase a full description distribution.
