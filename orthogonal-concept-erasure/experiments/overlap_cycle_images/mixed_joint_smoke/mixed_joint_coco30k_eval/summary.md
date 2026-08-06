# Mixed joint OCE — MSCOCO first-10k preservation

Prompt count: 10000

| Model | CLIP Score ↑ | FID to Original SD ↓ |
|---|---:|---:|
| Original SD | 31.1081 ± 2.8721 | 0.0000 |
| Mixed heterogeneous joint OCE | 30.5074 ± 3.0535 | 7.5452 |
| Difference (mixed − original) | -0.6007 | +7.5452 |

Original SD 是相同 prompt／seed 的 FID reference，因此其 FID 基準按定義為 0。

Metrics 成功後已刪除 Original 10000 張與 mixed 10000 張。中央 1k/10k Original CLIP baseline、FID statistics、protocol 與 prompt manifest 均已保留；per-image Inception features 未保留。
