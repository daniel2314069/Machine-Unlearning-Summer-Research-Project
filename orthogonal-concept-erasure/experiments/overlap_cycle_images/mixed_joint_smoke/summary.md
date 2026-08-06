# Mixed heterogeneous joint subspace smoke

This is a visual-only smoke test comparing Original SD with one joint model
edited by the repository's unchanged official OCE subspace objective.

## Actual joint set

| Domain | Target | Anchor | Generation prompt |
|---|---|---|---|
| Animal | cat | dog | `a photo of a cat` |
| Vehicle | truck | car | `a photo of a truck` |
| Building | church | castle | `a photo of a church` |
| Artist/style | Van Gogh | art | `Van Gogh style painting of the night sky with bold strokes.` |
| Celebrity | Adam Driver | celebrity | `a portrait of Adam Driver` |

Target set `{cat, truck, church, Van Gogh, Adam Driver}` and anchor set
`{dog, car, castle, art, celebrity}` have an empty intersection.

`Van Gogh` is the standard string in `trainscripts/style.sh`. `Adam Driver`
is the first target in the repository's official E10/E50/E100 celebrity
benchmark lists.

## Visual observations

| Target | Target roughly disappears? | Images roughly normal? | Obvious confusion or collapse? |
|---|---|---|---|
| cat | Yes | Yes | No broad collapse; all edited images are recognizable dogs. |
| truck | Yes | Mostly | Trucks disappear, but results are mostly scenery, buildings, surfaces, or textures rather than cars. This is the weakest pair. |
| church | Yes | Yes | Images become ordinary residential or office buildings rather than clear castles. |
| Van Gogh | Yes | Yes | The recognizable Van Gogh appearance changes into generic colorful or abstract starry art. |
| Adam Driver | Yes | Mostly | Adam Driver is no longer recognizable; outputs remain portraits but several faces are stylized or distorted. |

Overall, the mixed heterogeneous joint smoke does not show a global generation
collapse. Targets are broadly removed and most images remain readable, though
own-anchor behavior is inconsistent—especially truck → car. At this visual
smoke level, the run looks usable enough to justify later quantitative
evaluation rather than being obviously too broken to continue.

## Seed-aligned grids

### cat → dog

![cat](cat_mixed_joint_smoke.png)

### truck → car

![truck](truck_mixed_joint_smoke.png)

### church → castle

![church](church_mixed_joint_smoke.png)

### Van Gogh → art

![Van Gogh](van_gogh_mixed_joint_smoke.png)

### Adam Driver → celebrity

![Adam Driver](adam_driver_mixed_joint_smoke.png)
