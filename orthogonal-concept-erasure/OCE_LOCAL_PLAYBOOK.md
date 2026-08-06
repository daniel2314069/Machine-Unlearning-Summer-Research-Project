# OCE Local Playbook

Local copy of the official implementation for:

- Paper: Orthogonal Concept Erasure for Diffusion Models
- arXiv: https://arxiv.org/abs/2605.28902
- Upstream code: https://github.com/HansSunY/OCE
- Imported commit: `2609f3e` (`Rename`)

No license file was present in the upstream repository at import time.

## Setup

Use a separate environment from SPEED / ScaPre / DP unless you have already verified
the dependency versions are compatible.

```bash
cd orthogonal-concept-erasure
conda create -n oce python=3.10 -y
conda activate oce
pip install -r requirements.txt
```

The upstream requirements use `torch>=2.6`, `diffusers==0.33.1`, and
`transformers>=4.48.0`.

## First SD 1.4 Smoke Test

OCE requires the generic preservation term `Cg.pt` before SD erasure:

```bash
cd orthogonal-concept-erasure
python compute_Cg.py
```

Then run a small object erasure case:

```bash
bash trainscripts/object.sh
```

This default script erases `airplane` from `CompVis/stable-diffusion-v1-4` and
writes the edited model under `./airplane/`.

Generate a small image set:

```bash
python generate_object.py \
  --model_id "CompVis/stable-diffusion-v1-4" \
  --oce_model_path "airplane/airplane.safetensors" \
  --prompt "a photo of airplane" \
  --save_path "eval_airplane_smoke" \
  --exp_name "airplane" \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --device "cuda:0"
```

## GPU Notes

For SD 1.4 / SD 1.5, OCE should be a reasonable fit for an A4000 16GB setup.
The editing path in `oce.py` and `compute_Cg.py` uses `torch.float32`, which is
appropriate for the closed-form linear algebra. Do not switch the edit step to
fp16 for formal results unless you are explicitly testing the numerical impact.

For an 8GB GPU, use it only for smoke tests or generation with small sample
counts. If memory is tight, reduce generation sample count first. Avoid changing
the closed-form edit precision before changing evaluation workload.

The repository also includes FLUX scripts (`oce_flux.py`, `compute_Cg_flux.py`,
`trainscripts/flux_demo.sh`), but FLUX is intentionally out of scope for the
current local workflow.

## Batch Size / Sample Count

OCE's edit step is not SGD mini-batch training; the more relevant reproducibility
settings are concept lists, guide/preserve concepts, `Cg.pt`, edit scales, and
float precision.

For generation/evaluation, keep upstream sample counts when reproducing reported
numbers. For smoke tests, reduce only `--num_images_per_prompt` and record that
the run is not a formal comparison.
