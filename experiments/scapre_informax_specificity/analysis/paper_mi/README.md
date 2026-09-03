# ScaPre paper-MI comparison

This fixed experiment compares the existing repository implementation with the
paper-defined Informax weighting path while preserving the established
Confuse5 protocol.

The `paper_mi` arm changes only the MI data flow:

1. use the raw empirical binary MI for every target concept and output channel;
2. take the maximum across target concepts for each channel;
3. divide by the maximum aggregated channel to obtain `alpha`;
4. use `B = diag(alpha)` only in the final row-wise objective;
5. do not apply any per-concept MI weight in the earlier UCE accumulation.

The removed accumulation calls consume and discard the same two random tensors
without computing or applying MI. This keeps later aggregate pseudo-samples,
entropy samples, and all non-treatment RNG positions exactly paired with the
repository baseline while having no effect on the paper objective.

Prompts, five positive plus five negative pseudo-samples, per-channel median
threshold, strict `activation > threshold` binarization, official empty-string
neutral, evaluation, and edit seeds `20260820` through `20260824` remain fixed.
There is no parameter search. The formal repository baseline scores are reused
only after protocol, score, evaluator, historical source, and asset validation.
If the cleanup removed both the checksum-bound extracted cache and its archive,
the same run regenerates the repository baseline with the fixed protocol instead
of aborting. This fallback adds five repository edits and 15,000 generated
baseline images, but does not alter the comparison.

The experiment uses the project-established 25-class Confuse5 reconstruction.
It is suitable for the paired implementation comparison but is not an exact
reproduction of an author-released paper seed asset.

## Formal result

The completed fixed comparison and local validation are recorded in
[`formal_results/`](formal_results/README.md). Paper MI produced stronger
forgetting but substantially worse preservation and overall accuracy; its
max-normalized raw-MI alpha also saturated to an almost uniform vector.

## GPU server workflow

Run the end-to-end smoke check first:

```bash
cd /home/tslin/Documents/jupyter_data/anLi/machine_unlearning
conda activate MU
experiments/scapre_informax_specificity/analysis/paper_mi/run_server.sh smoke
```

Check it with:

```bash
experiments/scapre_informax_specificity/analysis/paper_mi/status_server.sh
```

After smoke reports `completed`, start the formal five-seed run:

```bash
experiments/scapre_informax_specificity/analysis/paper_mi/run_server.sh formal
```

Both launch modes use `nohup`, record PID/log/output/exit status, reject
duplicate active runs, and are safe across SSH disconnects. For a failed run,
`status_server.sh` prints its exact copyable resume command.

Asset preparation also runs inside the detached worker. Each launch validates
the pinned Conda packages, every recorded SD 1.5 component and size, the
ResNet50 checkpoint hash, and CUDA. If anything is absent or inconsistent, it
automatically invokes the repository setup/prefetch flow. Existing Hugging Face
and Torch caches are reused; missing model files are downloaded and the asset
manifest is rebuilt. The edited `official`/`paper_mi` checkpoints themselves are
always produced by the run and never need to be restored manually.

The worker automatically creates a lightweight verified archive in
`/home/tslin/Documents/jupyter_data/anLi/tmp`. Packaging can be checked or
retried after successful completion with:

```bash
experiments/scapre_informax_specificity/analysis/paper_mi/package_results.sh
```

Generated images and regenerable checkpoints are excluded from the archive.
The original server outputs, generated images, and checkpoints are preserved.

## Download to the Mac

From the repository root on the Mac, download the latest completed run through
the configured `tslin` SSH alias and verify its server-provided SHA-256:

```bash
experiments/scapre_informax_specificity/analysis/paper_mi/download_results.sh
```

The archive is saved under `~/Downloads/`. Neither the server archive nor the
original server outputs are deleted.

## Local static validation

This check uses only shell tools and does not run Python or load any model:

```bash
experiments/scapre_informax_specificity/analysis/paper_mi/validate_static.sh
```
