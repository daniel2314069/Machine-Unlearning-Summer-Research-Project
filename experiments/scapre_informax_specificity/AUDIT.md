# ScaPre Informax Specificity Audit

Audit date: 2026-08-20 (Asia/Taipei)

## Scope and provenance

This audit is intentionally completed before any ScaPre implementation change or
GPU experiment. The implementation authority is the checked-out repository; the
paper is used to identify the intended method and reported benchmark, not to
silently replace executable repository behavior.

- Workspace commit: `e195dc1db7a66f4bd717dda8aa46b04067624066`
- Workspace branch at audit time: `main` tracking `origin/main`
- Workspace status before this audit: clean
- ScaPre location: `scapre/` (a tracked directory in the workspace, not a Git
  submodule or independent checkout)
- ScaPre import commit in the workspace history:
  `f6e53043772c3469d85c3b22b0c5f6082ea0ed8f`
- Official upstream: <https://github.com/kaiyuan02415/ScaPre>
- Official upstream `main` audited at:
  `fb8bd1fac8ad25f7098d0b3b228c94a66cd9ccf8`
- Paper: *Forget Many, Forget Right: Scalable and Precise Concept Unlearning
  in Diffusion Models*, arXiv:2601.06162v4 / ICLR 2026
- The local `scapre/edit/erase.py`, evaluation scripts, datasets, and documented
  shell commands match upstream `fb8bd1f`. The local
  `scapre/edit/erase_scale.py` differs only by the already-existing
  `--output_model` option and its corresponding save path; the editing
  algorithm is unchanged relative to upstream.

Relevant file hashes at audit time:

| File | SHA-256 |
| --- | --- |
| `scapre/edit/erase.py` | `1ec6b1b8c116aaf3a075f2f4f60ec2864e3f5a22b5b5c11c113a13a83b9f9d92` |
| `scapre/edit/erase_scale.py` | `15d18f203e3619b160255257805c2daf27a87437a468fb1023d3f4e5c46e811f` |
| upstream `edit/erase_scale.py` | `9657344751378ff93bff532a885e0a0bede3c446315b297e076c0dd0f4c56836` |
| `scapre/eval/benchmarking/object_erase.py` | `e237294f5dbbdb1e41ffb3a9b3ded632b037d5341c5e6887a094c5f93a6e4465` |
| `scapre/eval/datasets/imagenet-15.csv` | `809b3ece6153ed3019e504cee1c50511f89f9cac5712a33bce16db09687ccb4d` |

## Stop-condition decision

**The hypothesis premise is valid under the current implementation.**

The current negative/reference input is not a pool of matched similar
non-target concepts. Both `scapre/edit/erase.py` and
`scapre/edit/erase_scale.py` compute an empty-string CLIP text embedding and use
that one vector, with independent Gaussian perturbations, as every Informax
negative example. Therefore the requested matched-retain intervention is not
already present, and the stated stop condition does not trigger.

## Informax implementation

### Entry points and functions

The complete ScaPre command documented for the Confuse benchmark is the second
command under `# Imagenet-Confuse15-15` in `scapre/script/erase.sh:71`, which
invokes `edit/erase_scale.py` with `--use_mi_softmask`, `--enable_ased`, and
`--bures_iters 1`.

Informax is implemented in:

- `scapre/edit/erase_scale.py:165`:
  `_compute_mi_softmask_emptyneg(...)`
- `scapre/edit/erase_scale.py:268`: `edit_model(...)`
- `scapre/edit/erase_scale.py:372-385`: per-concept Informax calls and
  max aggregation for `to_v`
- `scapre/edit/erase_scale.py:423-438`: per-concept Informax row weights in the
  `to_v` UCE accumulation
- `scapre/edit/erase_scale.py:492-505` and `542-556`: the corresponding two uses
  for `to_k`

`scapre/edit/erase.py` contains the same helper at lines 207-260 and uses it in
its separate non-scale editing path. It is not the full ScaPre path used by the
documented `erase_scale.py` Confuse command.

### Text features and reference input

In `edit_model`:

1. Target strings are tokenized by the Stable Diffusion pipeline tokenizer.
2. For each target, the positive base vector `c_vec` is selected as
   `emb[attention_mask.sum() - 2]`, i.e. the contextual CLIP text-encoder
   feature at the final content-token position (`erase_scale.py:329-334`). It
   is not an image, latent, or dataset sample.
3. The reference is tokenized from the literal empty string `""`. The code
   takes `blank_emb[0, 1, :]` as `empty_vec`
   (`erase_scale.py:335-340`). Thus the paper's `neutral input` is concretely
   an empty/null prompt embedding in this checkout, not a neutral concept and
   not a preserved concept.
4. The target replacement is a separate value: when no guided concept is
   supplied, `new_texts` contains a single-space string `" "`
   (`erase_scale.py:673-675`). This replacement must not be confused with the
   empty-string Informax negative.
5. `retain_text_`, `preserve_concepts`, and `preserve_scale` are parsed, but the
   resulting `ret_texts`/`preserve_scale` do not participate in the
   `erase_scale.py` editing equations. They are not Informax reference inputs.

### Positive and negative pseudo-samples

For every target concept and every Informax helper call:

- positives: five rows of `c_vec + epsilon`
- negatives: five rows of `empty_vec + epsilon`
- `epsilon`: independent element-wise Gaussian noise from `torch.randn`, scaled
  by the `noise_sigma` passed by `edit_model`
- current `edit_model` default and documented Confuse command value:
  `noise_sigma = 0.01`
- total: 10 pseudo-samples per target per helper call, with balanced binary
  labels (five positive, five negative)

The helper's standalone default is `noise_sigma=0.05`, but every relevant call
from `edit_model` explicitly passes the editing argument, whose default is
`0.01`. The helper parameter is named `num_pos`; the same value is also used as
the negative count. It is hard-coded as `5` at all Informax call sites.

The code does not set an editing RNG seed before these pseudo-samples. Therefore
two separately launched official runs are not guaranteed to use identical
Informax perturbations without an experiment-level seed control.

### Activation, threshold, binary state, and MI

For a layer weight matrix `W_old` and the 10 pseudo-samples:

1. `acts = W_old @ samples.T`, yielding one activation per output channel and
   pseudo-sample (`erase_scale.py:179`).
2. `tau_i` is the median of that channel's ten activations
   (`erase_scale.py:180`).
3. `z = 1{activation > tau_i}` uses a strict greater-than comparison
   (`erase_scale.py:181`).
4. The code counts the four binary `(z, y)` cells. It adds `eps=1e-8` to each
   cell count, then divides each by `K=10`; marginal probabilities are formed
   from those smoothed cell values (`erase_scale.py:182-197`).
5. It computes the four-term empirical discrete mutual information in natural
   logarithms.

### Normalization and soft weighting

The repository implementation does **not** transform MI as
`MI_i / max_j MI_j`. Instead, within each target and helper call it applies:

```text
mi_std_i = (mi_i - mean_channels(mi)) / (std_channels(mi) + eps)
softmask_i = sigmoid(mi_std_i / 0.7) ** p
```

For the documented full Confuse command, `p=8`. The returned tensor has shape
`(d_out, 1)` and is used as a soft row weight. In the solver it affects both:

- the per-concept UCE left-side increment `V`/`mat1_agg`; and
- the row-dependent concept-subspace term through
  `alpha_i = erase_scale * row_w_max[i]`.

Consequently, names such as `row_w`, `row_w_max`, and `alpha_i` in this
repository refer to the z-scored/sigmoid/powered soft mask and its scaled use,
not directly to the paper's max-normalized raw MI vector.

### Multi-concept aggregation

For each edited matrix (`to_v` and `to_k`) and layer:

1. The helper is called once per target concept.
2. The per-target soft masks are stacked.
3. `torch.max(..., dim=-1)` produces `row_w_max`, a channel-wise maximum over
   target concepts (`erase_scale.py:372-385`, `492-505`).
4. This maximum controls the row-specific concept-subspace term.
5. Inside the UCE accumulation loop, the helper is called again independently
   for each target, and that target's new stochastic soft mask weights its own
   `for_mat1` increment (`erase_scale.py:395-438`, `514-556`).

The second set of calls means the current implementation does not compute one
single immutable MI/alpha tensor and reuse it everywhere. A diagnostic must
distinguish at least the aggregate/subspace call from the per-concept
accumulation call; silently reusing one result would change official behavior.

### Current Informax data flow

```text
target strings
  -> tokenizer + CLIP text encoder
  -> final content-token vectors c_1 ... c_m

empty string
  -> tokenizer + CLIP text encoder
  -> empty_vec at token position 1

for every edited layer/matrix and target c_k:
  c_k + 5 Gaussian perturbations       (Y=1)
  empty_vec + 5 Gaussian perturbations (Y=0)
    -> W_old @ samples.T
    -> per-channel median threshold
    -> strict binary state z
    -> smoothed 2x2 empirical MI
    -> channel-wise z-score
    -> sigmoid(./0.7) ** p
    -> per-target row soft mask

per-target masks
  -> channel-wise max across targets
  -> row-dependent concept-subspace strength

fresh per-target masks
  -> weight each target's UCE mat1 increment

weighted UCE + concept projection + S/R regularization
  -> row-wise Cholesky solve
  -> Bures row proximal geometry alignment
  -> edited to_v / to_k matrices
```

## Official ImageNet-Confuse5 audit

### Paper-defined benchmark and metrics

The paper defines five groups with two targets and three visually similar
non-targets per group (10 targets and 15 retains total). Table 7 lists the same
25 concepts in the user specification, including `yawl` as a target and
`speedboat` as a retain. The spelling/capitalization in the paper is:

- `golden retriever`, `labrador retriever`, `german shepherd`,
  `Chesapeake Bay retriever`, `pug`
- `tabby`, `tiger cat`, `persian cat`, `Siamese cat`, `Egyptian cat`
- `orange`, `lemon`, `pomegranate`, `fig`, `Granny Smith`
- `yawl`, `lifeboat`, `speedboat`, `catamaran`, `schooner`
- `soccer ball`, `volleyball`, `tennis ball`, `rugby ball`,
  `ping-pong ball`

The paper defines:

- Unlearn Accuracy: residual classifier accuracy over the 10 target concepts
- Preserve Accuracy: classifier accuracy over the 15 similar retain concepts
- Overall Accuracy:
  `2 * (100 - A) * P / ((100 - A) + P)`

where `A` and `P` are percentage-valued Unlearn and Preserve Accuracy.

The paper states that object unlearning accuracy uses an ImageNet-pretrained
ResNet-50. It does not publish the Confuse5 prompt list, seed list, sampler,
steps, CFG, or per-concept image count in its experimental-settings text.

### Repository generation and classifier pipeline

The only public object evaluator is
`scapre/eval/benchmarking/object_erase.py`. Notably,
`eval_coco_clip.py` is byte-for-byte identical to `object_erase.py`; despite its
name and README command, it is not a COCO CLIP evaluator in this checkout.

`object_erase.py` performs the following:

- base model default: `runwayml/stable-diffusion-v1-5`
- edited UNet load: `torch.load`, then `pipe.unet.load_state_dict(...,
  strict=False)`
- generation dtype: `torch.float16`
- prompts and seeds: CSV columns `prompt` and `evaluation_seed`
- generation: `pipe(prompt).images[0]`, one image per row
- explicit seeding before each image: `torch.manual_seed(seed)` and
  `numpy.random.seed(seed)`
- sampler, inference steps, CFG, and resolution: not passed by the evaluator;
  they are inherited from the loaded Diffusers pipeline/model defaults
- classifier: `torchvision.models.resnet50` with
  `ResNet50_Weights.DEFAULT`, including that weight enum's preprocessing
- decision: classifier top-1 only
- correctness mapping: lowercase substring test
  `label in prediction or prediction in label`
- CSV rows are kept in file order and truncated to `--max_prompts`, whose
  default is 130

The public `imagenet-15.csv` contains 7,500 rows: exactly 500 rows for each of
15 concepts. Each class uses one prompt template, `an image of a {concept}`, and
has 500 stored evaluation seeds. The 15 present classes are:

```text
golden retriever, labrador retriever, german shepherd,
tabby, tiger cat, persian cat,
orange, lemon, pomegranate,
speedboat, lifeboat, yawl,
soccer ball, volleyball, tennis ball
```

The 10 paper benchmark classes missing from every public evaluation CSV are:

```text
Chesapeake Bay retriever, pug,
Siamese cat, Egyptian cat,
fig, Granny Smith,
catamaran, schooner,
rugby ball, ping-pong ball
```

### Can the current checkout reproduce the paper Confuse5 protocol?

**No.** The current public checkout cannot reproduce paper Table 7 as-is.

Concrete reasons:

1. The public dataset contains only 15 of the required 25 concepts and omits
   two retain classes from every group.
2. The paper's Table 7 values are quantized in increments consistent with
   120 images per concept (for example 80.8, 93.3, 74.2, and 2.5), while the
   public evaluator defaults to 130 and the CSV supplies 500. The 120-image
   interpretation is an inference from the reported values, not an explicitly
   published protocol field.
3. The documented shell section is labeled `Imagenet-Confuse15-15`, calls
   `erase/erase.py.py` in its first command (invalid path), and lists
   `speedboat` rather than paper-target `yawl`. Its second command uses the
   correct full ScaPre entry point but repeats the same target-list issue.
4. No public runner computes all 25 per-concept metrics, the five group metrics,
   or the aggregate Overall Accuracy.
5. The paper does not supply the missing prompt/seed records or pin the
   Diffusers generation defaults needed to reconstruct the exact images.
6. The purported COCO CLIP script is actually a duplicate of the object
   classifier evaluator, so this checkout cannot produce the formally named
   `CLIP_coco` result through that entry point.

This is a protocol-asset gap, not evidence against the Informax hypothesis.
It does mean that missing seeds or generation settings must not be silently
invented and described as the official paper protocol.

## Reproducibility risks to address before a formal run

- No Informax/editing RNG seed is set in the official editor.
- Official and matched variants must consume the same number and order of
  random draws; with five negatives and three retain concepts, the closest
  balance is `2/2/1`, with the extra assignments deterministically rotated or
  otherwise predeclared.
- The model identifier, model revision, tokenizer/text-encoder revision,
  scheduler configuration, Diffusers version, Torch/Torchvision versions, and
  ResNet weight identity must be resolved and recorded before generation.
- The existing `requirements.txt` pins `diffusers==0.35.0.dev0`; fresh-clone
  installation must verify that this exact distribution is obtainable or use
  an explicitly recorded compatible source/revision rather than silently
  drifting.
- Hugging Face model access and Torchvision ResNet weights are downloaded on
  first use. Fresh-clone server scripts must preflight authentication/cache,
  download them deliberately, and record resolved revisions/checksums where
  available.
- `load_state_dict(..., strict=False)` must be accompanied by an explicit
  missing/unexpected-key check in the experiment wrapper so an incompatible
  checkpoint cannot be evaluated as if valid.
- Existing generated images are reused solely by path. A formal wrapper must
  fingerprint checkpoint, prompt, seed, model revision, and generation config
  before reuse.

## Audit conclusion

The proposed ablation targets a real one-variable distinction in the current
implementation: empty-prompt negative bases versus matched similar-retain
negative bases, while retaining five total negatives per target.

Implementation may proceed without triggering the hypothesis-premise stop
condition. However, a claim of exact paper-protocol reproduction is blocked by
the absent 10-class Confuse5 evaluation asset and unpublished generation
details. The formal GPU run must wait until either:

1. the authors' complete 25-class prompt/seed protocol is obtained and hashed;
   or
2. a project-specific reconstruction is explicitly authorized and labeled as
   such, with fixed prompts/seeds and no claim that its absolute numbers
   reproduce paper Table 7.

## Post-audit project-asset resolution

A subsequent repository-wide search (beyond the `scapre/` checkout) found the
project's tracked 25-class asset at
`orthogonal-concept-erasure/experiments/confuse5_single_vs_joint/datasets/imagenet-confuse5-derived-25.csv`
(SHA-256 `f473503dd5a008f989a107e5adfe0749e9e2e77d8f613f2b7a4aae8bd87301d9`).
It was introduced by project commit `06ae690`, not by the ScaPre authors. Its
builder preserves the public 15-class CSV and constructs the ten absent retain
classes by reusing the ordered 500 seeds of the one available retain class in
the same group. The specificity experiment uses this established, hash-pinned
project reconstruction instead of inventing a second seed scheme. This closes
the project-level asset gap but does not change the audit conclusion that an
exact author-released Table 7 protocol is unavailable.

The official GitHub audit also covered both published branches (`main` and
`kaiyuan02415-patch-1`), the complete recursive `main` tree, tags, and releases.
The auxiliary branch contains only `LICENSE` and `README.md`; the repository
publishes no tags or release assets containing an additional Confuse5 dataset.
Accordingly, option 2 above is now implemented and the project-level run is not
blocked, while the exact-Table-7 caveat remains mandatory.
