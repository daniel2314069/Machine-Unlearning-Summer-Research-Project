# ScaPre Informax MI and channel-weighting diagnostics

This is an analysis subdirectory of the existing
`scapre_informax_specificity` experiment. It audits and measures the unchanged
official Informax calculation. It never edits model weights, calls the diffusion
pipeline, generates diffusion images, or invokes an image evaluator.

Order of execution is enforced by `run_diagnostics.py`:

1. reproduce the saved seed-20260820 aggregate-stage `n=5` diagnostic;
2. build one nested 50-positive/50-neutral pool per seed, projection, layer and
   target, then analyze prefixes `n=5,10,20,50`;
3. after sample-size outputs pass consistency checks, enumerate all 1,023
   non-empty subsets of the ten Confuse targets;
4. analyze the author-ordered ImageNet-Diversi50 list cumulatively at
   `m=1,5,10,20,30,40,50`.

The analysis RNG is a dedicated `torch.Generator` seeded once per requested
Informax seed. Iteration order is fixed (`to_v`, then `to_k`; layer index; target
order). Each observation draws tensors of shape `(50,d_in)` once for positives
and once for neutrals; smaller `n` always slices the first `n` rows. Thus nested
prefix equality does not depend on whether different-shape PyTorch draws happen
to share prefixes. This independent analysis stream intentionally does not
preserve later image-generation RNG positions because no image generation or
editing occurs.

On the GPU server:

```bash
conda activate MU
cd /path/to/Machine-Unlearning-Summer-Research-Project
experiments/scapre_informax_specificity/analysis/mi_channel_weighting/run_server.sh
experiments/scapre_informax_specificity/analysis/mi_channel_weighting/status_server.sh
experiments/scapre_informax_specificity/analysis/mi_channel_weighting/package_results.sh
```

If the prior artifact is not at the recorded default, pass its absolute path as
the first argument to `run_server.sh`. The optional second argument is an
absolute cached model snapshot; by default the launcher uses the exact resolved
snapshot recorded by the prior formal run. It validates every required
tokenizer, text-encoder, and UNet file before launch. The launcher checks Python syntax and
imports from the already-active `MU` environment; it never installs or downloads
packages. No local Mac Python execution is allowed. After packaging, the local
`download_results.sh` accepts either a configured SSH alias such as `tslin` or a
literal `user@host` destination.

The completed formal analysis and interpretation are in [summary.md](summary.md).
Checked-in result tables are under [results](results); the two largest CSVs are
stored as lossless `.csv.gz` files to remain below GitHub's single-file limit.
`finalize_retrieved_results.rb` performs only lightweight result-presentation
normalization and does not invoke Python, load a model, or rerun Informax.
