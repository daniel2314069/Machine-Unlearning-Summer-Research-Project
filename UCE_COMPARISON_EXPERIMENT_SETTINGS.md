# UCE Comparison Experiment Settings

This note consolidates the experiment settings used when comparing UCE with
SPEED, DP, ScaPre, OCE, FIA, SNCE, and SAEmnesia. It is a settings inventory,
not a new experiment report.

## Source Priority

| Source tier | How it is used |
|---|---|
| `paper/repo official` | Primary source. Use the official paper, official README, or official released scripts when available. |
| `local repo script` | Used only to fill implementation details that are missing from the official paper/README or to identify the local reproduction entry point. |
| `local modified experiment` | Not used for this comparison summary. In particular, the local ScaPre religious-buildings ASED experiment is excluded from the ScaPre settings below. |

## Common UCE Baseline

Official UCE for Stable Diffusion is a closed-form edit of cross-attention
weights. In the local UCE SD script, the edit targets cross-attention `attn2`
`to_v` and `to_k` modules and solves a regularized linear system from erase,
guide, and preserve text embeddings.

Common UCE hyperparameters are:

| Setting | Value |
|---|---|
| `erase_scale` | `1` |
| `preserve_scale` | `1` |
| `lamb` | `0.5` |
| Typical generation | `guidance_scale=7.5`, 50 inference steps |

For formal comparisons, use the generation and evaluation settings from each
method's comparison protocol rather than assuming these defaults everywhere.

Local references:

- `unified-concept-editing/trainscripts/uce_sd_erase.py`
- `unified-concept-editing/UCE_LOCAL_PLAYBOOK.md`
- `cvpr_double_projection/SD/UCE_original.py`

## SPEED

| Field | Setting |
|---|---|
| Method | SPEED |
| Compared UCE baseline | UCE is one of the reported baselines, alongside methods such as RECE, MACE, and Concept Ablation/ConAbl. |
| Base model | Stable Diffusion v1.4. |
| Task / benchmark | Few-concept erasure for instance and artist style; implicit nudity erasure on I2P; preservation on MS-COCO. |
| Targets | Instance concepts include Snoopy, Mickey, Spongebob, Pikachu, and Hello Kitty in the local few-concept script. Style concepts include Van Gogh, Picasso, Monet, Paul Gauguin, and Caravaggio. Nudity target is `nudity`. |
| Image count / prompt count | Paper setting: instance erasure uses 80 templates and artist style uses 30 templates, with 10 images per template per concept. Preservation uses the first 1,000 MS-COCO captions. I2P uses all 4,703 prompts. Local few-concept script uses `num_samples=10`; local I2P script uses `num_samples=1` per prompt. |
| Sampling setting | Paper setting: DPM-Solver, 20 sampling steps, CFG/guidance `7.5`. Local `speed/sample.py` defaults to `DPMSolverMultistepScheduler`, `total_timesteps=20`, `guidance_scale=7.5`, SD checkpoint `CompVis/stable-diffusion-v1-4`. |
| Metrics | Target erasure uses CLIP Score for few-concept tasks. Non-target/general preservation uses FID and CLIP Score. I2P nudity is evaluated with NudeNet at threshold `0.6`. |
| Important hyperparameters | Local SPEED few-concept script uses `params=V`, `aug_num=10`, `threshold=1e-1`. Local nudity script uses `retain_scale=0.70` and `lamb=0.5`. |
| Notes / caveats | Treat paper settings as authoritative for formal comparison. The local SPEED scripts are useful reproduction anchors, but this repo does not include a SPEED README. |

Local references:

- `speed/scripts/eval_few.sh`
- `speed/scripts/eval_nudity.sh`
- `speed/sample.py`
- `speed/data/i2p_benchmark.csv`
- `speed/data/mscoco.csv`

## DP / `cvpr_double_projection`

| Field | Setting |
|---|---|
| Method | DP / Double Projection, local folder `cvpr_double_projection`. |
| Compared UCE baseline | `cvpr_double_projection/SD/UCE_original.py`, described by the local README as a copy from the UCE official repo. |
| Base model | Stable Diffusion v1.4, `CompVis/stable-diffusion-v1-4`. |
| Task / benchmark | ImageNette-style 10-class object erasure. |
| Targets | The 10 classes in `small_imagenet_prompts.csv`: cassette player, chain saw, church, english springer, french horn, garbage truck, gas pump, golf ball, parachute, and tench. |
| Image count / prompt count | `small_imagenet_prompts.csv` has 100 prompts, 10 per class. Erased generation scripts use `num_variations=10`, giving 1,000 images per erased model evaluation. |
| Sampling setting | 512x512 resolution, 50 inference steps, guidance `7.5`, CSV `evaluation_seed`; local scripts use deterministic generation. |
| Metrics | `compare_accuracy.py` evaluates ResNet50 top-1 classification accuracy, comparing original vs. erased images. Logs report target accuracy drop and average drop over other classes. |
| Important hyperparameters | UCE baseline script uses `lamb=0.5`, `preserve_scale=1.0`, `erase_scale=1.0`. Per-target scripts guide erased objects to a related concept, for example cassette player to `box`, and preserve the other ImageNette classes. |
| Notes / caveats | The original-generation script default is `num_variations=20`, but the per-target erased DP/UCE scripts explicitly use `num_variations=10`; use the matched original/erased image sets required by the DP protocol. |

Local references:

- `cvpr_double_projection/SD/README.md`
- `cvpr_double_projection/SD/data/small_imagenet_prompts.csv`
- `cvpr_double_projection/SD/generate_original_sd.py`
- `cvpr_double_projection/SD/generate_erased_small_imagenet.py`
- `cvpr_double_projection/SD/compare_accuracy.py`
- `cvpr_double_projection/SD/scripts_SD14/UCE_CassettePlayer.sh`

## ScaPre

| Field | Setting |
|---|---|
| Method | ScaPre, official title "Forget Many, Forget Right: Scalable and Precise Concept Unlearning in Diffusion Models". |
| Compared UCE baseline | UCE is acknowledged as a baseline family in the official README; formal UCE numbers should come from the ScaPre paper tables. |
| Base model | Stable Diffusion v1.5 for the official repo commands. |
| Task / benchmark | Main official README example is Imagenette 10-class multi-object unlearning. Additional official tasks include large-scale object/style unlearning, explicit content unlearning, COCO quality, I2P, and artist erasure. |
| Targets | Imagenette object command erases `parachute, golf ball, garbage truck, cassette player, church, tench, english springer, french horn, chain saw, gas pump`. |
| Image count / prompt count | Use the paper's benchmark tables for formal counts. Repo evaluation entry points include Imagenette, I2P, artist erasure, and COCO CLIP Score. |
| Sampling setting | Evaluation scripts are task-specific. The README exposes the edit commands, not one universal generation count. |
| Metrics | Imagenette forgetting accuracy, generation quality/UQ, COCO CLIP Score, I2P NSFW evaluation, and artist CLIP similarity depending on task. |
| Important hyperparameters | Object command: `--base 1.5 --p 2 --alpha_min 0.8 --entropy_samples 20`. Large-scale object/style command: `--erase_scale 2 --p 8 --bures_iters 1 --enable_ased --entropy_samples 30 --entropy_bins 20`. Explicit command: `--erase_scale 2 --p 3 --bures_mu_from_entropy --bures_iters 1`. |
| Notes / caveats | The local religious-buildings ASED experiment is intentionally excluded. If the ScaPre paper and local scripts disagree, use the paper as the formal comparison source and the repo scripts as reproduction entry points. |

Local references:

- `scapre/README.md`
- `scapre/script/erase.sh`
- `scapre/script/eval.sh`
- `scapre/eval/benchmarking/object_erase.py`
- `scapre/eval/benchmarking/nudity_eval.py`
- `scapre/eval/benchmarking/artist_erasure.py`
- `scapre/eval/benchmarking/eval_coco_clip.py`

## OCE

| Field | Setting |
|---|---|
| Method | OCE / Orthogonal Concept Erasure. |
| Compared UCE baseline | Compare against SD UCE under the same task prompt, guide, preserve, generation, and metric protocol. |
| Base model | Main SD comparison uses Stable Diffusion v1.4, `CompVis/stable-diffusion-v1-4`. The repo also includes FLUX scripts, but SD is the primary comparison setting here. |
| Task / benchmark | Object erasure, style erasure, celebrity erasure, nudity/unsafe content erasure, COCO quality, CIFAR-10/object generation, I2P, and style/celebrity generation depending on script. |
| Targets | Object example erases `airplane`; style example erases `Van Gogh`; nudity example erases `nude; naked; sexual; impure; erotic; exposed; belly`; celebrity scripts erase 10/50/100 celebrity lists. |
| Image count / prompt count | Use upstream evaluation scripts and paper settings for formal counts. Local playbook smoke tests may reduce `num_images_per_prompt` and are not formal comparisons. |
| Sampling setting | SD generation smoke test uses 50 inference steps. Evaluation generation is task-specific under `orthogonal-concept-erasure/evalscripts`. |
| Metrics | FID, CLIP Score, celebrity detector, NudeNet, and CLIP classification accuracy, separated by task. |
| Important hyperparameters | `compute_Cg.py` must be run first to create `Cg.pt`. Object example: `erase_scale=2000`, `preserve_global_scale=10`, `preserve_concept_scale=0`, `lamb=10`, `expand_prompts=true`. Nudity example: guide/preserve `clothed; weared; fully dressed`, `erase_scale=3000`, `preserve_global_scale=20`, `preserve_concept_scale=0`, `lamb=10`, `expand_prompts=true`. |
| Notes / caveats | OCE is a multiplicative orthogonal/Procrustes-style edit using only `attn2.to_v`, not the additive UCE closed-form update. Do not compare raw hyperparameter magnitudes directly with UCE scales. |

Local references:

- `orthogonal-concept-erasure/OCE_LOCAL_PLAYBOOK.md`
- `orthogonal-concept-erasure/compute_Cg.py`
- `orthogonal-concept-erasure/oce.py`
- `orthogonal-concept-erasure/trainscripts/object.sh`
- `orthogonal-concept-erasure/trainscripts/nudity.sh`
- `orthogonal-concept-erasure/trainscripts/style.sh`

## FIA / `forget-it-all`

| Field | Setting |
|---|---|
| Method | FIA / Forget-It-All. |
| Compared UCE baseline | UCE must be run as a simultaneous multi-concept edit over the same concept set to be comparable with FIA's simultaneous multi-concept pruning. |
| Base model | Paper setting: object/art experiments mostly use SD1.5; explicit-content experiments use SD1.4 to align with prior baselines. Local README uses SD1.5 for COCO/object examples and SD1.4 for explicit evaluation. |
| Task / benchmark | Imagenette multi-object unlearning, I2P explicit content, multi-artist style, and MS COCO-30K preservation. |
| Targets | Object command erases the 10 Imagenette classes: `parachute, golf ball, garbage truck, cassette player, church, tench, english springer, french horn, chain saw, gas pump`. Art command erases `Van Gogh, Monet, Pablo Picasso, Leonardo Da Vinci, Salvador Dali`. Explicit command erases `naked`. |
| Image count / prompt count | Paper uses full task benchmarks. Local object evaluation uses `datasets/imagenette.csv`; explicit local evaluation can cap I2P with `--max_prompts 300`; COCO local reproduction can cap preservation with `--max_prompts 200`. |
| Sampling setting | Paper setting: all experiments use 50 denoising steps. |
| Metrics | Imagenette per-class forgetting accuracy, I2P explicit-content evaluation, artist evaluation, COCO CLIP Score, and generated-image sanity checks. |
| Important hyperparameters | Contrastive Concept Saliency uses 10 samples per concept in the paper; local configs use saliency steps of 10. Object config: `seed=1244`, `r1=0.05`, `r2=0.007`, `alpha=0.6`; harder concepts `parachute`, `golf ball`, and `gas pump` use `r2=0.03`. |
| Notes / caveats | FIA is a training-free neuron pruning/masking method with fused masks. Sequential single-concept UCE is not a fair baseline for FIA's simultaneous multi-concept setting. |

Local references:

- `forget-it-all/README.md`
- `forget-it-all/scripts/run_object.sh`
- `forget-it-all/scripts/run_explicit.sh`
- `forget-it-all/scripts/run_art.sh`
- `forget-it-all/evaluation/eval_object.py`
- `forget-it-all/evaluation/eval_explicit.py`
- `forget-it-all/evaluation/eval_coco.py`

## SNCE

| Field | Setting |
|---|---|
| Method | SNCE / A Single Neuron Works: Precise Concept Erasure in Text-to-Image Diffusion Models. |
| Compared UCE baseline | This local checkout does not include a full UCE reproduction harness for SNCE. UCE comparison tables should be taken from the SNCE paper or official benchmark release. |
| Base model | Stable Diffusion v1.4 CLIP text model SAE in the local README. |
| Task / benchmark | Demo/inference for safety concept erasure and steering. |
| Targets | Local README lists neuron data for nudity, knife, bloody/bleed, and gun. Example category is `naked`. |
| Image count / prompt count | Local code is a demo server/inference path, not a formal benchmark runner. |
| Sampling setting | Local app defaults should be treated as demo settings. The plan records demo settings of `steps=25` and `guidance_scale=7`; the README example uses `seed=2539888290`, `neruonum=10`, `strength=0.6`, `mode=steer`, and category `naked`. |
| Metrics | Not available from the local README. Use the SNCE paper/official evaluation code for formal UCE comparison metrics. |
| Important hyperparameters | `mode=steer`, `neruonum=10`, `strength=0.6`, category such as `naked`. Requires SD1.4 checkpoint, SAE checkpoint, and concept neuron data. |
| Notes / caveats | The local repo currently supports demo/inference confirmation. It should not be treated as a complete formal UCE comparison harness. |

Local references:

- `snce/README.MD`
- `snce/app.py`
- `snce/neuron_tool.py`

## SAEmnesia

| Field | Setting |
|---|---|
| Method | SAEmnesia / Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders. |
| Compared UCE baseline | UCE comparison must use the UnlearnCanvas protocol. Do not mix ImageNette accuracy or COCO CLIP tables into SAEmnesia comparisons. |
| Base model | UnlearnCanvas `style50` diffusion model, with supervised SAE assets. |
| Task / benchmark | UnlearnCanvas style50/object setup. |
| Targets | Implemented around UnlearnCanvas class/style combinations: 20 object classes by 50 styles plus seed-image prompts as generated by the scripts. |
| Image count / prompt count | The test structure is `20 object classes x 50 styles` plus seed-image prompts, as implemented by the sampling/evaluation scripts. |
| Sampling setting | `seed=188`, `steps=100`; plan records `guidance_scale=9.0`. Hookpoint is `unet.up_blocks.1.attentions.1`. |
| Metrics | UA, IRA, and CRA from the UnlearnCanvas evaluation pipeline; evaluation batch size `128`. |
| Important hyperparameters | Required assets include SAE checkpoint, `class_params.pth`, `cls_latents_dict_unet.up_blocks.1.attentions.1.pkl`, UnlearnCanvas `style50` diffusion model, and `style50.pth` / `style50_cls.pth` classifiers. |
| Notes / caveats | SAEmnesia is not an ImageNette/COCO protocol. Keep its UCE baseline, prompts, model, hookpoint, and classifiers aligned with UnlearnCanvas. |

Local references:

- `saemnesia/README.md`
- `saemnesia/scripts/sample_unlearning_cls_distr.py`
- `saemnesia/scripts/run_acc_all_cls.py`
- `saemnesia/scripts/accuracy_unlearncanvas_cls_fast.py`
- `saemnesia/UnlearnCanvas_resources/const.py`
