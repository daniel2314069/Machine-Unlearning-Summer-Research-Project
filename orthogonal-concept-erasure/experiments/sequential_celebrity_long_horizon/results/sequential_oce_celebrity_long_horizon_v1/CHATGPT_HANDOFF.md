# ChatGPT handoff: long-horizon sequential OCE celebrity experiment

## Ready-to-paste message

Please audit and interpret the completed long-horizon sequential OCE celebrity
experiment in this directory. The frozen protocol uses the official 100 target
celebrities in original and reverse order, grouped into ten legal 10-celebrity
OCE edits. Each order has independent `baseline` and `retain_history`
trajectories, plus one joint-100 reference.

The selected formal sampling protocol was `profile_5`. The core GCD experiment
is complete: 41 checkpoints/evaluation cells and 34,800 raw prediction rows.
The standalone integrity audit passed with the expected row count, zero
duplicate prediction keys, correct schedules and checkpoint parent chains,
paired samples, reproducible aggregates, and successful post-evaluation image
cleanup. MS-COCO was explicitly deferred by the Lightning budget and is not a
missing part of the core GCD result.

Please answer only the preregistered research question: whether repeated
sequential edits caused broad cumulative reappearance among celebrities that
were successfully erased at introduction, and, only if that baseline failure
exists, whether retaining the full erased history was a necessary systematic
improvement. Do not invent new hypotheses or metrics, cherry-pick celebrities,
reinterpret order effects as a new research question, or propose adding the
unused denser sampling profile after seeing the result.

Read `REPORT.md` first, verify integrity in `independent_audit.json` and
`final_validation.json`, then use `paper_checkpoint_results.csv`,
`step_summary.csv`, and `trajectory_per_concept.csv` for interpretation.
Consult `run_manifest.json` and the fixed input CSVs for protocol details. Use
`raw/all_gcd_predictions.csv` only when row-level verification is needed.

## Files to provide

### Minimum evidence set for a serious interpretation

1. `CHATGPT_HANDOFF.md`
2. `REPORT.md`
3. `independent_audit.json`
4. `final_validation.json`
5. `paper_checkpoint_results.csv`
6. `step_summary.csv`
7. `trajectory_per_concept.csv`

### Add for protocol audit or disputed calculations

- `run_manifest.json`
- `budget_selection.json`
- `inputs/target_schedule.csv`
- `inputs/retain_set.csv`
- `raw/all_gcd_predictions.csv` (34,800 row-level predictions; approximately
  25 MB)
- `events.jsonl` and `logs/` for execution troubleshooting

The 1.4 GB portable checkpoint archive and generated images are not required
for interpreting the completed result.
