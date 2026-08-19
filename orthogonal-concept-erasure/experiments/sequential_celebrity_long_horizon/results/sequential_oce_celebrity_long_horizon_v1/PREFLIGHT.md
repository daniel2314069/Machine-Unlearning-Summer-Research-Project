# Sequential OCE Celebrity Preflight

Status: protocol frozen; no model edit or image generation was launched.

## Design

- 100 repository E100 targets, batched as ten repository-E10-sized edits
- Order A is the repository list; Order B is its exact reverse
- baseline and retain-history each start independently from the same frozen SD snapshot
- retain-history at batch t appends exactly the prior 10 × (t - 1) targets
- the repository edit call reads each loaded parent checkpoint as its current pre-edit reference
- one repository joint-100 reference checkpoint
- no protocol-identical saved joint-100 result is assumed; the runner builds it once and resumes it by hash
- sequential repository guides: ['person', 'woman', 'man']
- sequential repository settings: {"dtype": "float32", "erase_scale": 3500.0, "expand_prompts": false, "guide_alignment_seed": 42, "guide_concepts": ["person", "woman", "man"], "lamb": 10.0, "preserve_concept_scale": 5.0, "preserve_global_scale": 50.0}
- joint repository settings: {"dtype": "float32", "erase_scale": 800.0, "expand_prompts": false, "guide_alignment_seed": 42, "guide_concepts": ["tree"], "lamb": 10.0, "preserve_concept_scale": 2.0, "preserve_global_scale": 70.0}

## Evaluation

- every sequential checkpoint: all introduced targets plus 500 fixed-retain images
- paired profile candidates: 5 or 10 trajectory images per celebrity/checkpoint
- 10/50/100 paper cells: 500 targets + 500 retains using official seed-42 sample streams
- joint-100: 500 targets + 500 retains
- nominal core generated totals (excluding benchmark): profile_5=34900, profile_10=45820
- profile and formal batch size are not chosen until the fixed 200-image generation + GCD benchmark with a 20% credit reserve

## Repository versus paper

- `celebrity_anchor`: paper=celebrity; repository=['person', 'woman', 'man']; authority=repository E10
- `retain_spelling`: paper=Melanie Grifftih; repository=Melanie Griffith; authority=repository string

## Frozen paths

- output: `/teamspace/studios/this_studio/runs/sequential_oce_celebrity_long_horizon_v1`
- artifact root: `/teamspace/studios/this_studio/artifacts/sequential_oce_celebrity_long_horizon_v1`
- future qualitative archive: `/teamspace/studios/this_studio/artifacts/sequential_oce_celebrity_long_horizon_v1/qualitative_samples.tar.gz`
- GCD project: `/teamspace/studios/this_studio/external/celeb-detection-oss`
- GCD resources: `/teamspace/studios/this_studio/external/celeb-detection-oss/examples/resources`

Exact ordered sets are in `inputs/target_schedule.csv` and `inputs/retain_set.csv`.
