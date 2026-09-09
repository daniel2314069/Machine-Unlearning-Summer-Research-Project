# ScaPre Informax Specificity 實驗總結

## 1. 實驗設計

ScaPre Informax 每個 target 使用 **5 positive + 5 negative pseudo-samples**。我們比較三個版本，唯一改變是 negative/reference source：

- **official**：empty prompt。
- **matched-retain**：同一 Confuse5 group 的三個 retain concepts；5 個 negatives 固定分配為 2/2/1。
- **superclass-neutral**：target 的 superclass；dogs=`dog`、cats=`cat`、fruits=`fruit`、boats=`boat`、balls=`ball`。

其餘全部固定：Stable Diffusion 1.5、10 targets、15 retains、generation prompts/seeds、sample counts、noise procedure、MI、threshold、alpha、ScaPre editing、sampler、50 steps、CFG 7.5、classifier 與 evaluator。

每個 variant/edit seed 評估：

`25 concepts × 120 images = 3,000 images`

指標：

- **Unlearn Accuracy ↓**：target residual accuracy，越低越好。
- **Preserve Accuracy ↑**：retain accuracy，越高越好。
- **Overall Accuracy ↑**：`2(100-U)P / ((100-U)+P)`。

## 2. 第一輪 matched-retain

Edit seed `20260820`：

| Variant | Unlearn ↓ | Preserve ↑ | Overall ↑ |
| --- | ---: | ---: | ---: |
| official | 19.75 | 40.61 | 53.93 |
| matched-retain | 17.42 | 41.72 | 55.44 |
| delta | -2.33 | +1.11 | +1.51 |

判定：**SUPPORTED**。

## 3. Matched-retain edit-seed robustness

Delta = matched-retain − official。

| Edit seed | ΔUnlearn ↓ | ΔPreserve ↑ | ΔOverall ↑ |
| --- | ---: | ---: | ---: |
| 20260820 | -2.33 | +1.11 | +1.51 |
| 20260821 | -1.75 | +0.39 | +0.73 |
| 20260822 | -2.50 | +1.61 | +1.99 |
| 20260823 | -1.75 | +0.56 | +0.88 |
| 20260824 | -2.25 | +1.44 | +1.78 |
| **Mean** | **-2.12** | **+1.02** | **+1.38** |

Preserve、Unlearn 與 Overall 均在 **5/5 seeds** 維持或改善。

判定：**ROBUSTLY SUPPORTED**。

## 4. Superclass-neutral

Delta = superclass-neutral − official。

| Edit seed | ΔUnlearn ↓ | ΔPreserve ↑ | ΔOverall ↑ |
| --- | ---: | ---: | ---: |
| 20260820 | -0.17 | +0.67 | +0.62 |
| 20260821 | +0.33 | +0.56 | +0.42 |
| 20260822 | -0.08 | +1.22 | +1.09 |
| 20260823 | -0.25 | -0.33 | -0.24 |
| 20260824 | 0.00 | +0.44 | +0.40 |
| **Mean** | **-0.03** | **+0.51** | **+0.46** |

Preserve 與 Overall 在 4/5 seeds 改善，但平均 Preserve gain 只有 `+0.51 pp`，低於事前門檻 `+1.0 pp`。

判定：**NOT SUPPORTED**。

## 5. 最終比較

| Variant（五-seed mean） | Unlearn ↓ | Preserve ↑ | Overall ↑ |
| --- | ---: | ---: | ---: |
| official | 19.27 | 40.46 | 53.90 |
| matched-retain | **17.15** | **41.48** | **55.28** |
| superclass-neutral | 19.23 | 40.97 | 54.36 |

## 6. Raw MI 與 α distribution

以下統計使用 edit seed `20260820` 的完整 `diagnostics/official.pt` 與 `diagnostics/matched_retain.pt`，涵蓋 `2 projections × 16 layers × 10 targets = 320` 個 records、每個 variant 共 `249,600` 個 channel observations。

### Raw MI 的離散值

因為 Informax 使用 5 positive、5 negative、median threshold 與 binary state，在沒有 activation ties 時，raw MI 主要只有三個可能值：

| Raw MI | 意義 |
| ---: | --- |
| 0.0201 | positive/negative 幾乎混合，弱區分 |
| 0.1927 | 中度區分 |
| 0.6931 | 完全區分，等於 `ln(2)` |

實際 channel 分布：

| Variant | 0.0201 | 0.1927 | 0.6931 |
| --- | ---: | ---: | ---: |
| official | 1,601（0.64%） | 2,096（0.84%） | 245,902（98.52%） |
| matched-retain | 82,775（33.16%） | 45,239（18.12%） | 121,586（48.71%） |

Official 另有 1 個 channel（0.0004%）為 `0.0863`；stored diagnostic 沒有保存該 channel 的完整 contingency table，因此不進一步推定原因。

Official 的 empty reference 使幾乎所有 channels 都得到最大 MI；matched-retain 則把大量共享語意 channels 移到低或中 MI。

### α distribution

這裡的 α 是 repository 實際使用的 soft mask：

`α = sigmoid(zscore(raw MI) / 0.7)^8`

它不是 paper 文字中的 `MI / max(MI)`。由於 z-score 在每個 layer/target record 內計算，相同 raw MI 在不同 records 不一定對應完全相同 α。

| Variant | Mean | Median | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official | 0.00743 | 0.00748 | 0.00858 | 0.00870 | 0.00920 | 0.01101 |
| matched-retain | 0.08865 | 0.00023 | 0.22601 | 0.25003 | 0.27855 | 0.41055 |

Matched-retain 的 α 呈現明顯分化：median 接近零，但高 MI channel 形成較大的上尾；official 因 raw MI 幾乎全部相同，α 集中在很窄的小範圍。

### Per-concept MI 與 α

下表將每個 target 的 32 個 projection/layer records 合併；數值是 channel-weighted mean。

| Target | Official MI | Matched MI | Official α | Matched α |
| --- | ---: | ---: | ---: | ---: |
| golden retriever | 0.6838 | 0.3399 | 0.00767 | 0.10058 |
| labrador retriever | 0.6846 | 0.3319 | 0.00747 | 0.10332 |
| tabby | 0.6852 | 0.4054 | 0.00726 | 0.08047 |
| tiger cat | 0.6839 | 0.3747 | 0.00761 | 0.09073 |
| orange | 0.6849 | 0.3942 | 0.00737 | 0.08369 |
| lemon | 0.6854 | 0.3862 | 0.00717 | 0.08630 |
| yawl | 0.6851 | 0.4335 | 0.00731 | 0.07207 |
| lifeboat | 0.6847 | 0.3698 | 0.00745 | 0.09258 |
| soccer ball | 0.6845 | 0.3271 | 0.00746 | 0.10441 |
| volleyball | 0.6841 | 0.4300 | 0.00755 | 0.07233 |

### 跨五個 edit seeds

320 個 layer/projection/target records 的 unweighted mean raw MI：

| Edit seed | Official MI | Matched MI | Superclass MI | Superclass mean α |
| --- | ---: | ---: | ---: | ---: |
| 20260820 | 0.68473 | 0.37960 | 0.68314 | 0.00782 |
| 20260821 | 0.68484 | 0.37964 | 0.68311 | 0.00784 |
| 20260822 | 0.68483 | 0.37973 | 0.68327 | 0.00781 |
| 20260823 | 0.68478 | 0.37975 | 0.68302 | 0.00785 |
| 20260824 | 0.68477 | 0.37967 | 0.68306 | 0.00785 |
| **Mean** | **0.68479** | **0.37968** | **0.68312** | **0.00783** |

Superclass-neutral 的 raw MI 幾乎與 official 相同；matched-retain 則跨五個 seeds 都維持明顯較低且更有區分度的 MI distribution。

完整數據另存於同一資料夾：

- `ScaPre_Informax_MI_alpha_per_layer_concept_seed20260820.csv`：320 個 per-layer/per-concept records，含 MI level proportions 與 α mean/std/quantiles。
- `ScaPre_Informax_MI_alpha_per_layer_seed20260820.csv`：32 個 projection/layer 彙總。
- `ScaPre_Informax_MI_alpha_per_concept_seed20260820.csv`：10 個 target 彙總。
- `ScaPre_Informax_MI_per_layer_concept_all_seeds.csv`：五 seeds 共 1,600 個 per-layer/per-concept mean-MI、α correlation 與 top-channel overlap records。
- `ScaPre_Informax_MI_alpha_distribution_seed20260820.json`：完整 distribution 統計與 counts。

## 結論

Empty prompt 並不是最理想的 Informax reference。使用 **matched retain concepts** 能跨五個 edit seeds 穩定提高 preservation，同時改善 target erasure；只改成 **superclass** 雖有小幅正向結果，但效果不足以支持為有效改進。

本結果使用 project-established ImageNet-Confuse5 reconstruction，不宣稱為 ScaPre paper Table 7 的完全重現。
