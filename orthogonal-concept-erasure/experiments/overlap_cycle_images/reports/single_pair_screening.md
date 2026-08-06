# Single-pair official OCE subspace screening

All grids use prompt `a photo of a {concept}`, seeds 42–51, 50 steps,
guidance 7.5, and 512×512 images. Descriptions below are visual observations
only.

| Mapping | Target disappears? | Looks like own anchor? | Short visual description |
|---|---|---|---|
| cat → dog | Yes | Yes, clearly | All ten edited images are recognizable dogs without obvious collapse. |
| dog → cat | Mostly | Partly | Several seeds show cats or larger felines, but some become unrelated people, streets, or other content. |
| horse → deer | Yes | No | Horses disappear, but results are mostly landscapes, unrelated animals, figures, or textures rather than deer. |
| dog → wolf | Yes | No | Dogs disappear, but results are unrelated animals, people, objects, or scenes rather than wolves. |
| wolf → cat | Yes | Partly | Many seeds become feline-looking animals, usually spotted or wild cats rather than clean domestic cats; a few are unrelated or stylized. |

## Grids

### cat → dog

![cat to dog](../single_pairs/cat_to_dog/cat_to_dog_single.png)

### dog → cat

![dog to cat](../single_pairs/dog_to_cat/dog_to_cat_single.png)

### horse → deer

![horse to deer](../single_pairs/horse_to_deer/horse_to_deer_single.png)

### dog → wolf

![dog to wolf](../single_pairs/dog_to_wolf/dog_to_wolf_single.png)

### wolf → cat

![wolf to cat](../single_pairs/wolf_to_cat/wolf_to_cat_single.png)
