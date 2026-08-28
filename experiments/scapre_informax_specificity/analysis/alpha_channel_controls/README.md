# ScaPre Informax alpha-channel controls

This experiment tests whether the final channel-to-alpha correspondence matters
while retaining the official empty-string neutral and every established
Confuse5 image/evaluator input.

Run smoke first on the GPU server:

```bash
cd /home/tslin/Documents/jupyter_data/anLi/machine_unlearning
conda activate MU
experiments/scapre_informax_specificity/analysis/alpha_channel_controls/run_server.sh smoke
```

After it completes and passes integrity checks, run formal:

```bash
cd /home/tslin/Documents/jupyter_data/anLi/machine_unlearning
conda activate MU
experiments/scapre_informax_specificity/analysis/alpha_channel_controls/run_server.sh formal
```

Status and packaging are single commands:

```bash
experiments/scapre_informax_specificity/analysis/alpha_channel_controls/status_server.sh
experiments/scapre_informax_specificity/analysis/alpha_channel_controls/package_results.sh
```

To inventory storage under the server project area before any manual cleanup,
run the read-only audit. It records directory totals, files above 100 MiB, all
`.pt` files, and a complete size-annotated tree in a timestamped text report
under `/home/tslin/Documents/jupyter_data/anLi/tmp`:

```bash
experiments/scapre_informax_specificity/analysis/alpha_channel_controls/audit_server_storage.sh
```

If formal reference validation aborts before launch, collect the archive,
manifest, historical-source, and compatibility hashes with the read-only
diagnostic:

```bash
experiments/scapre_informax_specificity/analysis/alpha_channel_controls/diagnose_official_reference.sh
```

The launcher is detached with `nohup`; closing SSH after its health check is
safe. Formal reuses the already verified official five-seed scores and generates
only the three controls (45,000 new images). The smoke stage additionally runs
two alternate shuffle salts and is never used for scientific conclusions.

`results/` is populated by the server run with the required CSV, JSON, and
`summary.md` deliverables. Regenerable checkpoints are hash-verified and removed
after their evaluation (or, for formal official, after diagnostics comparison),
with per-checkpoint cleanup records. Generated images stay out of the result
archive; their per-file SHA-256 manifest is archived, then the verified PNGs are
removed according to the established project practice. Raw scores, manifests,
diagnostics, logs, summaries, and cleanup evidence remain.

## Completed smoke

The verified seed-`20260820` smoke retrieval is summarized in
`smoke_results/analysis_notes.md`. Lightweight evidence is retained under
`smoke_results/`; the original `.pt` tensors and full server provenance remain
under `.local_artifacts/scapre_informax/`. Files selected for external GPT web
review are collected in `smoke_results/gpt_handoff/README.md`.

## Completed formal

The verified five-seed formal result and portable technical report are under
`formal_results/`. The full extracted archive, raw scores, image manifest,
Torch diagnostics, and source snapshots remain under
`.local_artifacts/scapre_informax/`.
