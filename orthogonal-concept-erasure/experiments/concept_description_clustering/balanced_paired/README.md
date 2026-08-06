# Balanced paired concept-description clustering

This isolated experiment controls semantic facet, effective SD 1.4 CLIP token length, shared generation slot, slot-level generation round, source/model label, and generation instructions. It does not modify previous datasets or results.

The mandatory primary condition is `cat`, `dog`, `fox`, and `bear` with 50 accepted descriptions each. The extension adds `wolf`, `rabbit`, `deer`, and `horse`, using the same 50 shared slots for 400 accepted descriptions. Both configurations use the exact previous fixed suffix and the unsuffixed EOT state only.

Run commands from `orthogonal-concept-erasure/experiments/concept_description_clustering`.

## Primary 4×50

Generate and SD 1.4-validate the resumable paired dataset:

```bash
./scripts/run_py310.sh -m concept_clustering.balanced_paired generate-validate --config balanced_paired/configs/primary_4x50.json --output balanced_paired/outputs/primary_4x50
```

Extract both representations and run spherical k-means:

```bash
./scripts/run_py310.sh -m concept_clustering.balanced_paired analyze --config balanced_paired/configs/primary_4x50.json --output balanced_paired/outputs/primary_4x50
```

Generate the report:

```bash
./scripts/run_py310.sh -m concept_clustering.balanced_paired report --config balanced_paired/configs/primary_4x50.json --output balanced_paired/outputs/primary_4x50
```

Run all stages:

```bash
./scripts/run_py310.sh -m concept_clustering.balanced_paired all --config balanced_paired/configs/primary_4x50.json --output balanced_paired/outputs/primary_4x50
```

Every generation image and score is checkpointed. Repeating the same command skips cached candidate/seed pairs and never replaces a frozen accepted `(slot_id, concept)` pair.

## Eight-animal 8×50 extension

```bash
./scripts/run_py310.sh -m concept_clustering.balanced_paired all --config balanced_paired/configs/secondary_8x50.json --output balanced_paired/outputs/secondary_8x50
```

## Original-W0 geometry analysis

The follow-up analysis reuses the cached final 8×50 dataset, loads only the
unchanged pretrained SD 1.4 model, and evaluates every original OCE-targeted
`attn2.to_v` projection separately. It does not generate descriptions or
images, load an edited checkpoint, or modify model weights.

```bash
./scripts/run_py310.sh -m concept_clustering.balanced_paired_w0_geometry --dataset balanced_paired/outputs/secondary_8x50 --model-id CompVis/stable-diffusion-v1-4 --oce-repo ../.. --output results/balanced_paired_w0_geometry --device cuda:0 --batch-size 32 --random-seed 314159 --force
```

The committed numerical outputs and concise report are under
`results/balanced_paired_w0_geometry/`.

## Revised W0 geometry with Euclidean controls

This revision reuses the cached 8×50 text/name embeddings, retains spherical
k-means as the primary condition, and adds raw and row-normalized ordinary
Euclidean k-means. It also reports every animal in Text and L0–L15 for all
three name-to-description readout conditions. Use a new output directory:

```bash
./scripts/run_py310.sh -m concept_clustering.balanced_paired_w0_geometry_revised --dataset balanced_paired/outputs/secondary_8x50 --embedding-cache results/balanced_paired_w0_geometry --model-id CompVis/stable-diffusion-v1-4 --oce-repo ../.. --output results/balanced_paired_w0_geometry_revised --device cuda:0 --batch-size 32 --random-seed 314159
```

## Smoke test

This checks two paired slots and one SD seed. It validates mechanics, not the formal three-seed inclusion criterion.

```bash
./scripts/run_py310.sh -m concept_clustering.balanced_paired smoke --config balanced_paired/configs/primary_4x50.json --output balanced_paired/outputs/smoke_primary --slots 2
```

## Design notes

- Slots use ten facets with the fixed `short, short, medium, long, long` schedule.
- Effective ranges are 14–19, 21–26, and 28–33 tokens, measured with the actual SD 1.4 tokenizer.
- `generation_round` belongs to the shared slot. `replenishment_round` records later attempts for missing entries without changing the controlled round.
- Accepted entries are selected only from candidates that pass the repository's existing SD 1.4 generation-validation criterion.
- The fixed suffix is exactly ` This sentence describes the concept`; the readout is its final `concept` token.
- EOT uses the unsuffixed description and `attention_mask.sum(dim=1) - 1`.
- Both representations are row-L2-normalized. Global centering is not used.
- Spherical k-means receives embeddings only. Labels are constructed after both fits complete.
