# Mixed joint OCE — first-1k preservation milestone

Prompt count: 1000

| Model | CLIP Score ↑ | FID to Original SD ↓ |
|---|---:|---:|
| Original SD | 31.1992 ± 2.8995 | 0.0000 |
| Mixed heterogeneous joint OCE | 30.5993 ± 3.1262 | 41.5806 |
| Difference (mixed − original) | -0.5999 | +41.5806 |

Original SD 是相同 prompt／seed 的 FID reference，因此其 FID 基準按定義為 0。

1k 圖片暫時保留，因同一程序將直接繼續生成至論文 first-10k；最終 10k metrics 成功後統一刪除。
