# ScaPre Informax superclass-neutral

Final controlled ablation in the existing ScaPre Informax specificity line.
It compares the verified five-seed `official` baseline against a new
`superclass_neutral` variant while changing only the Informax negative base.

The fixed mappings are in `config.json`; the implementation audit and exact
scope are in `AUDIT.md`. Formal baseline score rows are reused, not regenerated.
The formal run adds 15,000 superclass-neutral images and retains a predeclared
90-image paired qualitative set (30 each for official, matched-retain, and
superclass-neutral). Full evaluation images are deleted only after the result
archive has been created and checksum-verified; qualitative images are kept.

## Server use

```bash
conda activate MU
cd ~/Documents/jupyter_data/anLi/machine_unlearning

experiments/scapre_informax_specificity/superclass_neutral/run_server.sh smoke
experiments/scapre_informax_specificity/superclass_neutral/status_server.sh

# After smoke is completed:
experiments/scapre_informax_specificity/superclass_neutral/run_server.sh formal
experiments/scapre_informax_specificity/superclass_neutral/status_server.sh
```

Both launchers detach from SSH. No `jq`, package installation, model download,
or extra evaluator is introduced. Existing parent experiment assets are reused.

## Outputs

- `results/summary.md`
- `results/per_seed.csv`
- `results/per_group_seed.csv`
- `results/per_concept_seed.csv`
- `results/aggregate_across_seeds.csv`
- `results/per_group_robustness.csv`
- `results/per_target_robustness.csv`
- `results/per_retain_robustness.csv`
- `results/informax_seed_diagnostics.csv`
- `qualitative/images/<variant>/<group>/<concept>/...png`
- `qualitative/manifest.csv` and `qualitative/README.md`
- reproducibility manifests, raw score rows, actual configs, commands, and log

The final archive is written under
`/home/tslin/Documents/jupyter_data/anLi/tmp` and includes the small qualitative
set. `download_results.sh` prints the exact Mac-side `scp` and checksum commands.
