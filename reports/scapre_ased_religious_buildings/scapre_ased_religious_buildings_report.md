# ScaPre ASED R 宗教建築遺忘小實驗

## 實驗問題

本實驗比較 ScaPre 在遺忘一組宗教建築概念時，有無 ASED regularizer `R = U diag(tilde_sigma) U^T` 對圖片結果的影響。重點不是只看 target 是否被忘掉，而是看一般建築與宗教相關人事物是否被誤傷。

## 設定

- Base model: `runwayml/stable-diffusion-v1-5`
- 每類圖片數: `50`
- Inference steps: `50`
- Guidance scale: `7.5`
- 比較模型: `original_sd15`, `scapre_no_R`, `scapre_with_R`
- `with_R` 額外使用 `--enable_ased --T_sigma 1 --p_sigma 1`

## 指標摘要

`target_religious_buildings` 的 group/concept top-1 越低代表遺忘越強；其他三類越高代表 preserve 越好。`drift_clip` 越接近 1，代表越接近 original SD1.5 的同 prompt/seed 圖片。

### target_religious_buildings

| model | n | group top1 | concept top1 | prompt align | expected group score | drift clip |
|---|---:|---:|---:|---:|---:|---:|
| original_sd15 | 50 | 0.9600 | 0.7000 | 0.3048 | 0.2685 | - |
| scapre_no_R | 50 | 0.5000 | 0.2000 | 0.2557 | 0.2327 | 0.6194 |
| scapre_with_R | 50 | 0.4600 | 0.1600 | 0.2575 | 0.2325 | 0.6196 |

### near_religious_people_objects

| model | n | group top1 | concept top1 | prompt align | expected group score | drift clip |
|---|---:|---:|---:|---:|---:|---:|
| original_sd15 | 50 | 0.8800 | 0.8800 | 0.2950 | 0.2437 | - |
| scapre_no_R | 50 | 0.8000 | 0.6600 | 0.2655 | 0.2281 | 0.6766 |
| scapre_with_R | 50 | 0.8200 | 0.6800 | 0.2661 | 0.2289 | 0.6773 |

### near_generic_buildings

| model | n | group top1 | concept top1 | prompt align | expected group score | drift clip |
|---|---:|---:|---:|---:|---:|---:|
| original_sd15 | 50 | 0.5000 | 0.9400 | 0.2877 | 0.2182 | - |
| scapre_no_R | 50 | 0.3800 | 0.8800 | 0.2803 | 0.2088 | 0.7520 |
| scapre_with_R | 50 | 0.3800 | 0.7800 | 0.2796 | 0.2092 | 0.7511 |

### far_unrelated_objects

| model | n | group top1 | concept top1 | prompt align | expected group score | drift clip |
|---|---:|---:|---:|---:|---:|---:|
| original_sd15 | 50 | 0.7400 | 1.0000 | 0.2894 | 0.2259 | - |
| scapre_no_R | 50 | 0.9000 | 0.9600 | 0.2931 | 0.2289 | 0.8250 |
| scapre_with_R | 50 | 0.8600 | 0.9800 | 0.2942 | 0.2303 | 0.8186 |

## 本次結果觀察

- Target 宗教建築：with_R 的 group top1 比 no_R 低 4.00%，concept top1 低 4.00%；在這批樣本裡 R 沒有讓 erase 變弱，反而略強。
- 宗教人事物 preserve：with_R 的 concept top1 比 no_R 高 2.00%；這是目前最符合「R 減少宗教相關概念誤傷」的訊號。
- 一般建築 preserve：with_R 的 concept top1 比 no_R 低 10.00%；這表示本次設定下 R 沒有保護一般建築，甚至可能更傷一般建築細分類。
- 無關物件 preserve：with_R 的 concept top1 比 no_R 高 2.00%，差異很小，表示遠距概念大致穩定。
- 總結：這輪 50 張/類的小樣本支持「R 對宗教相關人事物有輕微保護」；但沒有支持「R 保護一般建築」，一般建築的 concept top1 反而下降。

## 初步判讀方式

- 如果 `scapre_with_R` 在 target 類別和 `scapre_no_R` 一樣低，但 near preserve 類別更高，代表 R 有幫助。
- 如果 `scapre_with_R` 的 target 類別明顯高於 `scapre_no_R`，代表 R 可能過度保守。
- 如果 `scapre_no_R` 在 `near_generic_buildings` 掉很多，代表不加 R 可能傷到建築共享方向。
- 如果 `scapre_no_R` 在 `near_religious_people_objects` 掉很多，代表不加 R 可能傷到宗教共享方向。

## 限制

- 這是小樣本實驗，CLIP 分數只能做方向性判斷。
- 這裡的 drift 是 generated-to-generated 比較，不是真實資料 FID。
- 最終仍需要人工看 grid，確認建築細節、宗教符號與圖片品質是否合理。
