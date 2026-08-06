# ScaPre ASED R 宗教建築實驗說明

這個實驗比較 ScaPre 在遺忘一組宗教建築時，有無 ASED regularizer
`R = U diag(tilde_sigma) U^T` 對圖片結果的影響。

主控腳本：

```bash
python scapre/experiments/run_ased_religious_building_ablation.py
```

## 建議跑法

先在 ScaPre 環境中跑小量 smoke test：

```bash
python scapre/experiments/run_ased_religious_building_ablation.py \
  --mode all \
  --max-images-per-group 4 \
  --num-inference-steps 10
```

正式小批次：

```bash
python scapre/experiments/run_ased_religious_building_ablation.py \
  --mode all \
  --max-images-per-group 50 \
  --num-inference-steps 50
```

如果模型編輯階段已經完成，只重跑圖片與評分：

```bash
python scapre/experiments/run_ased_religious_building_ablation.py \
  --mode eval \
  --max-images-per-group 50
```

如果只想重產中文報告與 grid：

```bash
python scapre/experiments/run_ased_religious_building_ablation.py \
  --mode report
```

## 低容量策略

腳本每次只保留一個小批次的完整圖片：

1. 產生 `1 model x 1 prompt group` 的圖片。
2. 立即用 CLIP 評分。
3. 抽樣複製少量圖片到 `kept_samples/`。
4. 刪除完整批次圖片。

預設不保留完整中間圖片。除錯時才使用：

```bash
--skip-delete
```

## 輸出位置

預設輸出在：

```text
reports/scapre_ased_religious_buildings/
```

重要檔案：

- `checkpoints/scapre_no_R.pt`
- `checkpoints/scapre_with_R.pt`
- `metrics/scores.csv`
- `metrics/summary.json`
- `kept_samples/`
- `grids/*.png`
- `scapre_ased_religious_buildings_report.md`

## 指標解讀

- `target_religious_buildings` 的 top-1 越低，代表宗教建築遺忘越強。
- `near_religious_people_objects` 越高，代表宗教相關人事物越沒有被誤傷。
- `near_generic_buildings` 越高，代表一般建築越沒有被誤傷。
- `drift_clip_cosine_to_original` 越接近 1，代表越接近 original SD1.5 的同 prompt/seed 圖片。

如果 `scapre_with_R` 的 target erasure 接近 `scapre_no_R`，但 near-preserve 分數更高，
就支持 R 有保護相近概念的效果。

## 注意

ScaPre 的 edit step 會用 SD1.5 並做 fp32 權重編輯，8GB GPU 可能不穩。
如果本機 OOM，建議在 A4000 16GB 上先跑 `--mode edit`，再把 checkpoints 帶回本機跑小量 eval。
