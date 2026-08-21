# ScaPre Informax Edit-Seed Robustness Audit

## Repository and prior-result provenance

The audit started from clean `main` at
`e15714052f470a89acf2673c8477b5bbd990ca6a` after
`git pull --ff-only origin main` reported that the checkout was current. That
commit contains the completed one-seed formal result.

Controlled sources before this robustness implementation:

| Source | SHA-256 |
| --- | --- |
| `scapre/edit/erase_scale.py` | `537fce48480ca8fa2233e28a3526c81ab6f9d848a055b317dea1eebb89724d33` |
| parent `config.json` | `3ce815fe4bdaca02370e77e7d7d27b8db226e99638824a10ede3b7528341cddc` |
| `evaluate_confuse5.py` | `6e5063bc709e6ecbc543e86058f91686bbf517298cf3fc3191f566f333f69ed8` |
| `build_protocol.py` | `539a0636c6e9185173aac509930710c7b7bebd0f24fa69a32fa11346df96aac6` |
| derived 25-class protocol source | `f473503dd5a008f989a107e5adfe0749e9e2e77d8f613f2b7a4aae8bd87301d9` |

The editor still exposes exactly `official` and `matched-retain`. The existing
empty-negative and matched-negative functions, five-positive/five-negative
sample counts, 2/2/1 matched assignment, threshold, empirical MI, alpha
transformation, maximum aggregation, spectral/geometry code, and solver were
not modified for this experiment.

## Seed 20260820 integrity

The downloaded archive
`scapre_informax_specificity_formal_20260820T163033Z_20260821T055813Z.tar.gz`
was verified against SHA-256
`35fde6c41d150146ef3022e830a772a547afb3234d731fa4acf03f9c9d09e4d1`.
Its paths were checked before extraction.

Independent checks over the raw score files confirmed:

- 3,000 rows per variant, with 1,200 targets and 1,800 retains;
- identical 3,000 generation keys between variants and no duplicate key;
- official U/P/O = `19.75 / 40.611111 / 53.930361`;
- matched U/P/O = `17.416667 / 41.722222 / 55.436946`;
- controlled-ablation status `passed` and judgment `SUPPORTED`.

The legacy run manifest says its working tree was dirty. This cannot be
silently treated as clean. Reuse is allowed here only because the archive
preserves the controlled sources and their hashes, those hashes match the
required parent commit, and the raw scores independently reproduce every
published metric. The new robustness run itself requires an empty start and
end `git status`; any new unexplained dirty state aborts the run.

## Why `--edit-seed` cannot simply be changed

The legacy CLI seeds Python, NumPy, CPU Torch, and CUDA Torch globally immediately
before `edit_model`. Inside `edit_model`, the same Torch stream is consumed by:

1. `_compute_entropy_factor`, which influences row entropy and `mu`; and
2. `_compute_mi_softmask_emptyneg` or `_compute_mi_softmask_matchedneg`, which
   creates the 5+5 Informax pseudo-samples.

Changing only the existing `--edit-seed` would therefore also change
non-Informax entropy samples and downstream geometry. That would violate the
requested controlled design.

## Informax-only RNG isolation

`informax_seed_runner.py` runs the byte-unchanged editor and intercepts only
direct `torch.randn` calls made by the two Informax functions. For every such
call it:

1. consumes and discards the draw from the legacy global stream, preserving the
   exact global RNG position used by later non-Informax operations; and
2. returns an identically shaped draw from an Informax-only generator seeded
   with the requested new seed.

All `_compute_entropy_factor` calls continue to use the legacy global stream
seeded at `20260820`. Official and matched-retain use the same Informax seed,
call order, tensor shapes, and number of draws. A formal ten-target edit must
intercept exactly 1,280 `randn` calls per variant; smoke must intercept 256.
The run aborts if counts or shape histograms differ.

This wrapper is seed-control instrumentation. It does not alter negative bases,
the MI estimator, weighting, solver, or any ScaPre method source. Default parent
experiment behavior is untouched because the parent runner never invokes the
wrapper.

## Generation and evaluator isolation

The parent protocol builder produces one protocol CSV for the entire run. Its
formal hash must equal the seed-20260820 protocol hash
`ab8ced3be82b1d8c6896eb24d32419bb12e8cf3e22d8ed0c46f92f4a5e8f0f2e`.
Every variant and edit seed is evaluated against that same file. The aggregator
compares all `(group, role, concept, sample_index, prompt, generation seed,
seed source)` keys across all ten variant/seed combinations and rejects any
difference. Evaluator manifests must also match after excluding only variant
name and checkpoint hash.

## Cleanup boundary

Generated PNGs are deleted only after aggregation, integrity validation, and a
verified result archive. Formal cleanup covers the four new seed runs and the
two explicitly named previous smoke/formal runs. It never deletes scores,
manifests, diagnostics, checkpoints, archives, caches, or model weights.
