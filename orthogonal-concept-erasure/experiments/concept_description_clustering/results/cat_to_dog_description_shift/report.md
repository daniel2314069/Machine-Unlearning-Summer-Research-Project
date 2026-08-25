# Cat-to-Dog OCE Shift of Name-Free Animal Descriptions

## 1. Research Question

Does the verified OCE `cat -> dog` transformation move the 50 name-free cat descriptions toward the original dog anchor while leaving the seven non-target animal classes relatively unchanged? This is the only question tested.

## 2. Cat -> Dog OCE Setup

The analysis uses the existing verified float32 checkpoint for SD 1.4 with edit concept `cat`, guide concept `dog`, preserve concept `dog`, prompt expansion `true`, erase scale 2000, global-preservation scale 10, concept-preservation scale 0, and lambda 10. The checkpoint contains exactly the 16 `attn2.to_v` tensors targeted by the repository's `Orthogonal_Erase` implementation. It is read as tensors and is not loaded into a generation pipeline.

## 3. Fixed Original Cat and Dog References

At each layer, the cat and dog references are `W0_l c_cat` and `W0_l c_dog`, where the bare-name vectors use the repository's exact final-content-token-before-EOT rule. Both references are L2-normalized and remain fixed in the original W0 coordinate system. Edited descriptions are deliberately not compared with `Wcat_l c_dog`: doing so would move the reference and confound description movement with anchor movement.

## 4. Description-Level Delta Metrics

Only cached unsuffixed EOT description vectors are used. `delta_dog = cosine(Wcat h, W0 c_dog) - cosine(W0 h, W0 c_dog)`; positive values mean movement toward the original dog direction. `delta_cat = cosine(Wcat h, W0 c_cat) - cosine(W0 h, W0 c_cat)`; negative values mean movement away from the original cat direction. All four cosine inputs are L2-normalized within the same layer. No clustering is rerun because this experiment asks only about before/after directional change.

## 5. Canonical Cat Sanity Check

The bare canonical cat vector moves with positive canonical delta_dog and negative canonical delta_cat in 0/16 analyzed layers. Per-layer values are retained in `canonical_cat_sanity_check.csv`; this check is separate from the 400 description results.

## 6. Results for Name-Free Cat Descriptions

| Layer | Cat mean delta_dog | Cat mean delta_cat | Cat joint fraction | Non-target mean delta_dog | Non-target mean delta_cat |
|---|---|---|---|---|---|
| L0 | +0.0161 | -0.0298 | 0.20 | +0.0492 | +0.0605 |
| L1 | +0.0192 | -0.0097 | 0.20 | +0.0191 | +0.0065 |
| L2 | -0.1161 | -0.2259 | 0.00 | -0.0370 | -0.0360 |
| L3 | +0.0061 | -0.0276 | 0.20 | +0.0572 | +0.1005 |
| L4 | -0.1071 | -0.3425 | 0.00 | -0.0240 | -0.0484 |
| L5 | -0.0930 | -0.3259 | 0.00 | -0.0298 | -0.0651 |
| L6 | -0.0840 | -0.3187 | 0.00 | -0.0179 | -0.0299 |
| L7 | -0.0921 | -0.4063 | 0.00 | -0.0216 | -0.0583 |
| L8 | -0.0811 | -0.3718 | 0.00 | -0.0123 | -0.0379 |
| L9 | -0.0256 | -0.1579 | 0.04 | +0.0231 | +0.0586 |
| L10 | -0.1286 | -0.4322 | 0.00 | -0.0340 | -0.0496 |
| L11 | -0.1215 | -0.3354 | 0.00 | -0.0522 | -0.1138 |
| L12 | +0.0669 | +0.0898 | 0.02 | +0.0392 | +0.0527 |
| L13 | +0.0790 | +0.1105 | 0.04 | +0.0876 | +0.1295 |
| L14 | +0.0037 | +0.0235 | 0.00 | +0.0060 | +0.0380 |
| L15 | -0.0826 | -0.3162 | 0.00 | -0.0106 | -0.0325 |

The cat-only plot shows the mean and one-standard-deviation spread across 50 descriptions, plus the joint intended-direction fraction. It preserves the two requested deltas rather than replacing them with another shift score.

## 7. Non-Target Specificity

The non-target controls are dog, fox, bear, wolf, rabbit, deer, and horse, each with 50 descriptions. The comparison plot contrasts the cat-description mean delta_dog with the equally weighted mean of the seven class-level non-target means. Detailed class-by-layer statistics remain in `class_layer_shift_summary.csv`.

## 8. Layer-Wise Results

The heatmaps retain all eight classes and every analyzed layer. The first two show signed class means; the third shows the fraction satisfying both `delta_dog > 0` and `delta_cat < 0`.

## 9. Main Interpretation

The canonical bare-name cat vector moves in the intended direction in 0/16 layers. For name-free cat descriptions, mean delta_dog is positive in 6/16 layers, mean delta_cat is negative in 13/16 layers, and the joint fraction exceeds 0.5 in 0/16 layers. Under the specified fixed-original-anchor metric, the answer is no: the canonical sanity check fails in every layer, cat-description dogward movement is not consistent, and the largest cat joint fraction is only 0.20. None of Cases A-D fully applies because each presumes a usable canonical cat-to-dog directional effect or a consistent description-level pattern. This classification is descriptive and remains layer-specific.

## 10. Limitations

This experiment measures directional geometry in original per-layer cross-attention output coordinates. It does not generate images and therefore does not prove image-level erasure or replacement. The existing checkpoint uses the repository's recorded object-edit prompt expansion in addition to the canonical bare names, so the result characterizes that verified `cat -> dog` edit rather than a hypothetical bare-name-only checkpoint. Results are descriptive and may vary across layers; vectors from different layers are never compared directly.
