# Final study conclusion

## Decision

**Stop the current AFR method branch after `AFR-I0`. Do not promote it to the
formal 500/class, dogs/fruits, COCO, CLIP, or FID stages.**

This is a negative method result, not a failed or corrupted experiment. The
matrix diagnostics passed their numerical and provenance checks; the proposed
image-level mechanism failed its preregistered smoke criterion.

## What the study established

1. Released QR rank inflation was not the main defect in the qualified
   Confuse5 matrices.
2. Objective-faithful OCE orientation improved the paper objective but left
   high transformed-target leakage.
3. Exact orthogonal target-to-anchor subspace mapping was feasible in all
   48/48 layers.
4. Within the exact orthogonal family, jointly keeping anchor features and the
   frozen preservation geometry cheap showed a substantial trade-off.
5. The non-orthogonal F/G editors satisfied their matrix guarantees. Full AFR
   G was nontrivial and reduced frozen-S distortion relative to pure projection
   F in all 48/48 layers.

## Why the proposed editor stops here

On the frozen 100/class balls smoke, objective-faithful C had target residual
macro accuracy `0.075`. Pure projection F and full AFR G increased it to
`0.505` and `0.480`, respectively. Both also moved matched-anchor probability
in the wrong direction. Thus exact edited-layer residual removal did not
produce the intended image-level target erasure or literal target-to-anchor
behavior.

G nevertheless showed a real preservation benefit over F: similar non-target
macro accuracy improved from `0.750` to `0.8133`, and mean fixed-seed anchor
LPIPS decreased from `0.1437` to `0.1094`. This validates the compensation
ablation but does not rescue the failed editing mechanism.

## Final classification

**AFR-I0 — Matrix success, image smoke fails.**

The defensible contribution from this branch is the OCE solver/subspace
diagnosis, the exact-orthogonal trade-off controls, and the negative result
showing that edited-layer subspace alignment alone is not sufficient for the
desired generative behavior. No claim should describe the current AFR editor
as successful.
