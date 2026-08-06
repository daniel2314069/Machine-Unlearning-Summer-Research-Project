# OCE overlap/cycle image-only summary

This run used only the unchanged official OCE subspace objective and
seed-aligned image grids. No CLIP metrics, LPIPS, feature matrices,
permutation checks, confusion matrices, heatmaps, or vector-wise comparisons
were executed.

## 1. Single-pair screening

- Clearly clean: **cat → dog**.
- Partly anchor-like but not consistently clean: **dog → cat**, **wolf → cat**.
- Not visually clean: **horse → deer**, **dog → wolf**.

Full visual notes and grids:
[single_pair_screening.md](single_pair_screening.md).

## 2. 2-cycle

Executed despite the single-pair gate, as explicitly requested.

For both cat and dog prompts, the joint-subspace column looks visually
indistinguishable or nearly indistinguishable from Original SD. The single
cat → dog model produces dogs, and the single dog → cat model often produces
felines, but the joint 2-cycle does not visibly express those two changes.

### Cat prompt

![2-cycle cat](../cycle2/cycle2_cat_prompt.png)

### Dog prompt

![2-cycle dog](../cycle2/cycle2_dog_prompt.png)

## 3. 3-cycle

Executed despite the single-pair gate.

For cat, dog, and wolf prompts, the joint-subspace column visually remains
at the original prompted concept. The requested cat → dog → wolf → cat cycle
is not visibly expressed in the joint images.

### Cat prompt

![3-cycle cat](../cycle3/cycle3_cat_prompt.png)

### Dog prompt

![3-cycle dog](../cycle3/cycle3_dog_prompt.png)

### Wolf prompt

![3-cycle wolf](../cycle3/cycle3_wolf_prompt.png)

## 4. No-overlap

Executed despite horse → deer being visually unclean as a single pair.

- Cat prompt: joint subspace produces recognizable dogs, broadly similar to
  the single cat → dog result.
- Horse prompt: horses disappear, but the joint images mostly show dogs
  rather than deer.

### Cat prompt

![no-overlap cat](../no_overlap/no_overlap_cat_prompt.png)

### Horse prompt

![no-overlap horse](../no_overlap/no_overlap_horse_prompt.png)

## 5. Blocked experiments

None. All requested experiments were executed because the user's opening
instruction explicitly overrode the later visual-cleanliness gates.
