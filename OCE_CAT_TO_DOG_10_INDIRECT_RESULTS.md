# OCE cat → dog：10 個不含關鍵字的描述測試

## 結論

在 10 組成對測試中，修改後權重 `W` 呈現強烈的 cat → dog 方向性：cat/dog 二分類的平均 dog 機率由 `W_0` 的 23.35% 上升至 `W` 的 85.98%。加入 mouse、rabbit、fox 後的五類檢查中，`W_0` 有 6/10 張以 cat 為最高類別，`W` 則有 8/10 張以 dog 為最高類別。

這次結果支持 OCE 修改確實會被「間接描述」觸發，而不是只有 prompt 明寫 `cat` 才生效。但它不是乾淨的語意替換：部分圖片沒有生成主要動物、生成玩偶或人類，也有修改後形體失真的案例。因此 CLIP 上的 dog 提升不能單獨解讀成 10/10 都成功生成正常的狗。

## 十個 prompts

以下句子均未使用 `cat`、`kitten` 或 `feline`：

1. A graceful four-legged house pet with triangular ears, bright eyes, long whiskers, and a slender swishing tail.
2. A soft-coated lap companion curled into a tight circle, with alert pointed ears and delicate whiskers.
3. A tiny indoor hunter crouching beside a toy mouse, its flexible tail raised and its paws ready to pounce.
4. A fluffy household companion sitting on a windowsill, watching birds with upright ears and twitching whiskers.
5. A playful little pet batting a ball of yarn across the floor with one paw while its long tail curves behind it.
6. A quiet night-roaming house pet with reflective eyes, padded paws, sharp retractable claws, and a balancing tail.
7. A sleek four-legged companion grooming its fur with its tongue, then rubbing its cheek against the furniture.
8. A small whiskered pet stretching its front paws after waking from a nap in a warm patch of sunlight.
9. An agile indoor creature climbing onto a bookshelf, guided by keen eyes, pointed ears, and a long flexible tail.
10. A purring lap-sized companion kneading a soft blanket with its front paws, eyes half closed in contentment.

## 聚合結果

| 指標 | `W_0` 原始權重 | `W` OCE 權重 |
|---|---:|---:|
| 平均 cat 機率，cat/dog 二分類 | 76.65% | 14.02% |
| 平均 dog 機率，cat/dog 二分類 | 23.35% | 85.98% |
| 平均 cat 機率，五分類 | 65.04% | 8.11% |
| 平均 dog 機率，五分類 | 17.80% | 70.72% |
| 五分類最高類別為 cat | 6/10 | 1/10 |
| 五分類最高類別為 dog | 1/10 | 8/10 |

五分類候選為 cat、dog、mouse、rabbit、fox，評估模型為 `openai/clip-vit-base-patch32`。

## 視覺觀察

- Prompts 1、2、4、8 最清楚：`W_0` 生成貓，`W` 生成狗。
- Prompt 7 的 `W_0` 本身已生成狗，因此不能用來證明轉換，但 `W` 仍維持狗。
- Prompt 3 的 `W` 生成玩偶；prompt 6 的 `W` 有嚴重動物形體失真。
- Prompts 5、9、10 沒有穩定生成主要動物，CLIP 類別分數的解釋力較弱。
- 因此較保守的結論是：間接描述可觸發強烈的 cat → dog 偏移，但伴隨生成品質與語意遵循副作用。

## 實驗控制

- `W_0` 與 `W` 每個 prompt 使用相同 seed（42–51）
- Stable Diffusion v1.4、512 × 512、50 inference steps、CFG 7.5
- 每個 prompt、每個權重各生成 1 張，共 20 張
- OCE 權重與前一次實驗相同，未重新訓練

## 產物

- 最外層比較圖：`OCE_cat_to_dog_10_prompts_W0_vs_W.png`
- 逐張圖片：`oce_cat_to_dog_comparison/indirect_10_results/W_0/` 與 `W/`
- 逐張 CLIP 分數：`oce_cat_to_dog_comparison/indirect_10_results/clip_scores.csv`
- 聚合數值：`oce_cat_to_dog_comparison/indirect_10_results/summary.json`
- Prompt 清單：`oce_cat_to_dog_comparison/prompts_10_indirect.txt`
