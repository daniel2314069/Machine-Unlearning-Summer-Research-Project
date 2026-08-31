# Server storage cleanup review — 2026-08-31

## 結論

這次 inventory 涵蓋的 repository root 目前占用 **104.76 GiB**。專案收尾時，建議分兩層清理：

1. **先刪 ScaPre 的 8 個 generated `runs/` 根目錄：91.013 GiB。** 這些路徑內沒有任何 Git-tracked file；正式 metrics、integrity reports、manifests、diagnostics 與三個 projection variant 的已下載 archive 都已有獨立保存。
2. **確定不再續跑 OCE 後，再刪 OCE 的未追蹤 raw generated images 與 binary model/checkpoint files：11.787 GiB。** 自動 cleanup 只接受位於 `images/` 或 `generated_images/` path component 下的圖片；contact/review sheets、plots、grids 與獨立 qualitative figures 保守保留。不要刪整個 `outputs/`，因為其中仍混有應保留的 results、reports、JSON manifests 與少量未追蹤輕量產物。

兩層合計可回收 **102.800 GiB**，repository root 預估剩下 **1.957 GiB**。這個估計使用 inventory 的 charged allocated bytes，且兩層候選互不重疊。

本報告只做 read-only 判讀；尚未刪除任何 server 或 Mac 檔案。

## 證據與完整性

- 掃描根目錄：`/home/tslin/Documents/jupyter_data/anLi/machine_unlearning`
- Inventory run：`inventory_20260831T114812Z`
- Archive：`machine_unlearning_storage_inventory_20260831T114812Z_20260831T114848Z.tar.gz`
- Archive SHA-256：`ccc15ab2bfa0e4c10b42b51972960d7c07c491ee4c6bb26ee0962dbc9c69a145`
- Archive、sidecar checksum、tar structure、23-file package manifest 與 11-file result manifest 均已驗證。
- 掃描完成狀態：passed，exit code 0，scan errors 0。
- 掃描數量：67,398 files、9,413 directories、3 symlinks；symlinks 未被 follow。
- Server Git：commit `db6f41a0abc45ed772b1ebb346b2a37665e9804b`、branch `main`、working tree clean。
- Logical file size：104.58 GiB；charged allocated size（包含 directory metadata）：104.76 GiB。
- 未發現 hardlink duplicate，因此候選容量沒有因 hardlink 被重複計算。

## 第一層：建議刪除 ScaPre generated runs

以下 8 個根目錄互不重疊，合計 **97,724,227,584 allocated bytes（91.012779 GiB）**、46,650 files。`git ls-files` 交叉檢查結果為 **0 tracked files**。

| 路徑 | Files | Allocated GiB |
| --- | ---: | ---: |
| `experiments/scapre_informax_specificity/runs` | 116 | 12.829803 |
| `experiments/scapre_informax_specificity/seed_robustness/runs` | 310 | 38.496334 |
| `experiments/scapre_informax_specificity/superclass_neutral/runs` | 334 | 19.338371 |
| `experiments/scapre_informax_specificity/analysis/mi_channel_weighting/runs` | 74 | 0.292629 |
| `experiments/scapre_informax_specificity/analysis/alpha_channel_controls/runs` | 384 | 0.171932 |
| `experiments/scapre_informax_specificity/analysis/projection_accumulation/runs` | 15,141 | 6.507240 |
| `experiments/scapre_informax_specificity/analysis/projection_accumulation_direct_cos2/runs` | 15,144 | 6.682869 |
| `experiments/scapre_informax_specificity/analysis/projection_accumulation_budget_matched_cos2/runs` | 15,147 | 6.693600 |

刪除理由：

- 主要空間是可重新產生的 `.pt` checkpoints 與 generated images。
- Original specificity、seed robustness、superclass neutral、alpha controls、projection V1、direct cos² 與 budget-matched cos² 的正式結果均已有 repository 內的 results/formal-results 保存；抽查的 integrity reports 都是 `passed`。
- 三個 projection formal archives 已下載回 Mac；package 本來就刻意排除 generated images 與 regenerable checkpoints。
- `mi_channel_weighting` 的 251.59 MiB raw CSV 已有 tracked gzip result，無須保留 run copy。

刪除後果：exact generated pixels 與 checkpoint bytes 無法由輕量 archive 直接還原，只能依 protocol 重新生成。這不影響已保存的正式 metrics、hashes、manifests 與結論；如果仍想人工查看原始圖片，應在清理前另選少量 review images，而不是保留整批 runs。

## 第二層：OCE generated images 與 weights

OCE 的 source、tracked result tables 與 reports 要保留；只考慮未追蹤的 raw generated images 與真正的 binary weights。下表合計 **12,655,955,968 allocated bytes（11.786777 GiB）**、13,879 files。Inventory 原始 category 另含 87 個 checkpoint-directory JSON manifests；它們不是 binary weights，因此已從刪除候選排除。另有 780 張約 0.698 GiB 的 contact/review sheets、plots、grids 與獨立 qualitative figures，不納入自動刪除。

| Experiment | Artifact | Files | Allocated GiB |
| --- | --- | ---: | ---: |
| `concept_description_clustering` | raw images | 12,407 | 6.013901 |
| `confuse5_single_vs_joint` | binary weights | 34 | 1.180614 |
| `sequential_object_pair_retain` | binary weights | 30 | 1.071350 |
| `correspondence_diagnostic` | checkpoints / weights | 26 | 1.012062 |
| `sequential_object_persistence` | binary weights | 20 | 0.714188 |
| `sequential_object_followup` | binary weights | 15 | 0.535641 |
| `oce_failure_image_qualification` | binary weights | 10 | 0.357094 |
| `overlap_cycle_images` | checkpoints / weights | 9 | 0.321384 |
| `correspondence_diagnostic` | raw images | 530 | 0.231373 |
| `overlap_cycle_images` | raw images | 430 | 0.194431 |
| `confuse5_single_vs_joint` | raw images | 368 | 0.154739 |

其中：

- 自動刪除的未追蹤 raw images 共 **6.594444 GiB**、13,735 files。
- 未追蹤 binary weights 共 **5.192333 GiB**、144 files；副檔名只限 `.safetensors/.pt/.pth/.ckpt`。
- 這些 artifacts 不在 Git；刪除不會弄髒 working tree，但會失去免重跑的圖片與 edited weights。
- `concept_description_clustering` 的分析資料、embeddings、metrics 與 reports 已 tracked；其約 6.014 GiB raw `generated_images/` 是最優先的 OCE 圖片候選，review/plot 類圖片則保留。
- `oce_failure_image_qualification` 已有 project-provided `package_review_images.sh` 與 fail-closed `cleanup_server_images.sh`。若尚未保存其固定規則 review set，應先使用既有 packaging workflow，再刪 images。

不要以 `rm -rf orthogonal-concept-erasure/experiments/*/outputs` 之類的廣泛命令清理。OCE outputs 內同時混有正式資料，必須由 cleanup script 依 Git tracking、category 與 allowlist 精確選取。

## 必須保留

1. **OCE shared evaluation references（約 0.092 GiB）**

   完整保留 `orthogonal-concept-erasure/experiments/evaluation_references/`。Registry 中 SD1.4 first-1k 與 first-10k references 都是 `complete`，包含 Original CLIP baseline、FID statistics、ordered prompt manifest 與 protocol fingerprint。刪除後會違反 project 的 reference reuse contract，並導致不必要的 Original baseline 重跑。

2. **所有 Git-tracked source、configs、results、reports、manifests 與 raw score tables**

   不要以整個 experiment directory 為單位刪除。特別保留 ScaPre `results/`、`reproducibility/`、各 `formal_results/`，以及 OCE 已追蹤的 `outputs/` 內 metrics/predictions/manifests。

3. **OCE 尚未封裝的輕量未追蹤 outputs（0.214039 GiB）**

   除 images/weights 外，inventory 還有 2,130 個未追蹤 OCE files，共 229,822,464 allocated bytes。容量很小，其中包括 Confuse5 result/evaluation files、logs、metadata 與 run outputs；在建立輕量 archive 或確認已被 tracked results 完整涵蓋前先保留。

4. **`.git/`（約 0.360 GiB）**

   不要手動刪除 objects 或 pack files。相較 103.5 GiB generated artifacts，其空間收益很小；若日後真的需要縮減，只能使用正常 Git maintenance，而不是直接刪 `.git/objects`。

5. **本次 inventory archive、sidecar checksum 與本報告**

   它們是清理前的可稽核快照。至少保留到清理後 inventory 驗證完成。

## 建議執行順序

1. 保持 server working tree clean，記錄 cleanup 前 commit。
2. 先以 allowlist cleanup script 處理上述 8 個 ScaPre `runs/` roots；不得使用 unresolved glob。
3. 重新跑 storage inventory，確認約回收 91.013 GiB，且 `git status --porcelain` 仍為空。
4. 若確認 OCE 不再續跑，先保存任何仍想人工 review 的小型 image selection，再由第二個 allowlist stage 只刪未追蹤 images 與 weights。
5. 再次跑 inventory，驗證 shared references、tracked results 與 manifests 仍存在，並保存 cleanup manifest。

清理前預估 104.76 GiB；只清 ScaPre 後預估 **13.744 GiB**；兩層都清理後預估 **1.957 GiB**。實際 filesystem 回收量可能因 directory blocks 與清理後 metadata 略有差異，因此正式 cleanup 必須以前後 inventory 為準。

## 範圍限制

這份 inventory 只掃描 `machine_unlearning` repository root。Server 上相鄰的 `/home/tslin/Documents/jupyter_data/anLi/tmp`、Conda environments、Hugging Face/model caches 與其他 home-directory 資料不在本次範圍內。尤其 `anLi/tmp` 可能仍有多份已下載的 experiment archives；在確認 Mac checksum 與備份策略後，可以另做 inventory，但不能由本報告推定其可刪容量。
