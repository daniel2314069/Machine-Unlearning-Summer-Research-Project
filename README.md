# Machine-Unlearning Summer Research Project

Research code, experiment configurations, evaluation records, and analysis
artifacts for studying concept erasure and machine unlearning in text-to-image
diffusion models. This is an independent research workspace and is not an
official repository of the methods listed below.

## Methods and provenance

- `orthogonal-concept-erasure/`: an imported and locally extended copy of
  [Orthogonal Concept Erasure (OCE)](https://github.com/HansSunY/OCE), the
  official implementation accompanying *Orthogonal Concept Erasure for
  Diffusion Models* ([arXiv:2605.28902](https://arxiv.org/abs/2605.28902),
  ICML 2026 Oral). Local experiment and evaluation guidance is documented in
  `orthogonal-concept-erasure/OCE_LOCAL_PLAYBOOK.md`.
- `scapre/`: an imported copy of
  [ScaPre](https://github.com/kaiyuan02415/ScaPre), accompanying *Forget Many,
  Forget Right: Scalable and Precise Concept Unlearning in Diffusion Models*
  ([arXiv:2601.06162](https://arxiv.org/abs/2601.06162), ICLR 2026), together
  with local research integration where applicable.
- `speed/`: SPEED experiments and evaluation scripts.
- `cvpr_double_projection/`: Double Projection / DP implementations and
  experiments.
- `unified-concept-editing/`: Unified Concept Editing (UCE) implementation and
  experiments.

## Attribution and licensing

Third-party code remains the property of its respective authors and is subject
to the license, notices, and terms supplied by each upstream project. Inclusion
in this repository does not relicense third-party material.

- ScaPre is distributed under the MIT License. Its upstream copyright and
  license notice are preserved in `scapre/LICENSE`.
- The OCE upstream repository did not include a software license file when its
  licensing status was checked on 2026-08-06. Accordingly, no additional
  permission or license for the upstream OCE code is asserted here; all rights
  to that upstream material remain with its authors.
- Other imported methods, datasets, model components, and evaluation tools are
  governed by their own upstream terms. Consult the relevant subdirectory and
  upstream source before redistribution or reuse.
- Project-specific scripts, configurations, reports, and AI-assisted materials
  that do not carry a separate license are publicly viewable in this repository,
  but no additional open-source license is granted by default.

Please cite the corresponding papers and upstream repositories when using or
building on a method. This repository does not distribute Stable Diffusion
checkpoints or grant rights to any separately downloaded model weights or
datasets.
