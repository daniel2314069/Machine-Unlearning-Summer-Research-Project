# OCE zero-generation qualification audit

This experiment audits the unchanged official `oce.py` implementation. It
does not run the UNet forward pass, generate images, or save edited model
weights.

It reproduces the official single-object protocol in `trainscripts/object.sh`
for `airplane`, `bird`, `dog`, and `truck`. Direction 1 partitions the final
official `R - I` using the head count read from each runtime `attn2` object.
Direction 3 records the exact upstream determinant branch and tests numerical
null-space SVD freedom with CPU float64.

On tslin, update the project checkout, activate the existing GPU environment,
and launch the background audit from the repository:

```bash
cd orthogonal-concept-erasure
conda activate MU
bash experiments/zero_generation_qualification_audit/run_server.sh
```

Check progress:

```bash
bash experiments/zero_generation_qualification_audit/status_server.sh
```

The full audit remains under this experiment's `outputs/` directory. On
success, the worker automatically places a small return archive containing the
report and tabular evidence in:

```text
/home/tslin/Documents/jupyter_data/anLi/tmp
```

The final `scp` of that archive to the Mac `~/Downloads` directory is manual.
