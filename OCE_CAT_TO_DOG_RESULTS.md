# OCE cat → dog 實驗結果

## 結論

在相同 prompt、seed 與推論設定下，OCE 編輯後的輸出在 CLIP cat/dog 二分類中由偏向 cat 轉為偏向 dog：dog 機率從 29.29% 上升至 76.79%。方向性轉換成功，但修改後影像以人眼觀察仍帶有鼠類／混種動物特徵，並非乾淨、典型的狗。

## 測試 prompt

> A small furry household pet with pointed ears, whiskers, and a flexible tail.

## 對照結果

| 權重 | CLIP cat | CLIP dog | 判定 |
|---|---:|---:|---|
| `W_0`（原始 SD 1.4） | 70.71% | 29.29% | cat |
| `W`（OCE cat → dog） | 23.21% | 76.79% | dog |

CLIP 數值使用 `openai/clip-vit-base-patch32`，候選文字為 `a photo of a cat` 與 `a photo of a dog`，表中為兩候選間 softmax 機率。這是單一 seed 的配對測試，不應視為整體成功率。

## OCE 設定

- Base model: `CompVis/stable-diffusion-v1-4`
- Edit concept: `cat`
- Guide concept: `dog`
- Preserve concept: `dog`
- Prompt expansion: `true`
- Erase scale: `2000`
- Global preserve scale: `10`
- Concept preserve scale: `0`
- Lambda: `10`
- 修改層：UNet 的 16 個 `attn2.to_v` 權重
- 編輯精度：float32
- `C_g`：COCO-30k 全部 30,000 prompts，共 403,727 個有效 token

## 生成設定

- Seed: `42`（`W_0` 與 `W` 相同）
- Resolution: 512 × 512
- Inference steps: `50`
- Guidance scale: `7.5`
- Generation dtype: bfloat16
- 每組圖片數：1

## 產物

- 最外層比較圖：`OCE_cat_to_dog_W0_vs_W.png`
- 原始圖：`oce_cat_to_dog_comparison/W_0.png`
- 修改後圖：`oce_cat_to_dog_comparison/W.png`
- OCE 權重：`oce_cat_to_dog_comparison/oce_cat_to_dog_W.safetensors`

