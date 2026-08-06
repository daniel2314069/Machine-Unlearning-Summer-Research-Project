#!/usr/bin/env python
"""Build the deterministic 1,500-row name-free candidate corpus.

The grammar is deliberately balanced: five concept-specific subject paraphrases
are crossed with three facet-specific predicates for every concept/facet group.
The normal validator remains authoritative and must be run on the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FACETS = {
    "visual_appearance": [
        "stands fully visible under daylight",
        "poses against a plain background",
        "appears in detailed side profile",
    ],
    "body_parts": [
        "displays its distinctive anatomy clearly",
        "leans closer for facial detail",
        "raises one forelimb beside torso",
    ],
    "movement": [
        "runs across open ground mid-stride",
        "changes direction with balanced limbs",
        "crosses a clearing in motion",
    ],
    "behavior_habits": [
        "performs a familiar daily habit",
        "pauses during a typical routine",
        "investigates its surroundings attentively",
    ],
    "sounds": [
        "opens its mouth mid-call",
        "vocalizes toward the camera",
        "holds a recognizable sound-making pose",
    ],
    "habitat_scene": [
        "occupies its typical natural environment",
        "appears within its ordinary habitat",
        "stands among its preferred surroundings",
    ],
    "human_interaction": [
        "stands beside a gentle person",
        "responds to an outstretched hand",
        "shares a scene with caretaker",
    ],
    "function_ability": [
        "demonstrates a characteristic physical ability",
        "uses its specialized body naturally",
        "shows a familiar practical skill",
    ],
    "object_relationship": [
        "interacts with familiar nearby objects",
        "stands among familiar daily items",
        "touches an associated object nearby",
    ],
    "indirect_holistic": [
        "anchors a story through posture",
        "forms one silhouette among objects",
        "appears centrally among identifying clues",
    ],
}


CONCEPTS = {
    "cat": [
        "A small purring house pet with pointed ears, whiskers, paws, and curling tail",
        "A furry windowsill companion with triangular ears, long whiskers, and retractable claws",
        "A soft-coated indoor pet with triangular ears, facial whiskers, padded paws, and curling tail",
        "A graceful purring companion with round eyes, pointed ears, whiskers, and silent paws",
        "A compact domestic pet with rough tongue, retractable claws, whiskers, and flexible tail",
    ],
    "dog": [
        "A barking household pet with wet nose, collar, leash, and wagging tail",
        "A loyal fetching companion with floppy ears, broad muzzle, and strong paws",
        "An energetic tennis-ball chaser wearing a collar above its furry body",
        "A friendly long-snouted pet with panting mouth, expressive ears, and moving tail",
        "A vigilant domestic companion with leather collar, sturdy legs, and wagging tail",
    ],
    "rabbit": [
        "A tiny white furry carrot eater with long upright ears, twitching nose, and cotton tail",
        "A small white burrow pet with long upright ears, buck teeth, carrots, and round tail",
        "A fluffy Easter garden visitor with long ears, twitching nose, carrots, and cotton tail",
        "A compact white carrot nibbler with tall ears, buck teeth, soft paws, and round tail",
        "A gentle burrow dweller with white fur, long ears, twitching nose, and cotton tail",
    ],
    "fox": [
        "A slender red-orange woodland prowler with black paws, pointed muzzle, and white-tipped bushy tail",
        "A lean red-orange woodland creature bearing triangular ears, dark feet, sharp snout, and pale-tipped tail",
        "A lean red-orange woodland hunter with black legs, triangular ears, sharp muzzle, and white-tipped tail",
        "A slender russet forest hunter with black feet, pointed ears, narrow muzzle, and white-tipped tail",
        "A red-orange woodland hunter with lean body, dark paws, sharp muzzle, and white-tipped bushy tail",
    ],
    "bear": [
        "A hibernating brown cave giant with round ears, broad muzzle, short tail, and clawed paws",
        "An enormous shaggy forest giant standing upright with round ears, broad nose, and heavy paws",
        "A massive hibernating woodland giant with round ears, broad muzzle, short tail, and clawed paws",
        "An enormous brown cave forager standing upright with round ears, broad nose, and heavy paws",
        "A heavy brown mountain giant with rounded face, broad muzzle, short tail, and clawed feet",
    ],
    "horse": [
        "A tall riding mount with saddle, flowing mane, long face, and four hard hooves",
        "A long-legged stable dweller with bridle, silky mane, muscular shoulders, and hooves",
        "A graceful pasture runner with arched neck, sweeping tail, mane, and slender legs",
        "A strong saddled mount with upright ears, long muzzle, flowing mane, and hooves",
        "A high-shouldered farm companion with bridle, long face, mane, and polished hooves",
    ],
    "cow": [
        "A black-and-white dairy grazer with broad muzzle, visible udder, horns, and cloven hooves",
        "A heavy pasture milker with spotted hide, four teats, neck bell, and tufted tail",
        "A broad barnyard grazer with square muzzle, short horns, udder, and black patches",
        "A slow grass-chewing farm creature with barrel torso, neck bell, and visible udder",
        "A sturdy dairy field dweller with wide nostrils, rounded flanks, horns, and teats",
    ],
    "sheep": [
        "A wool-covered flock grazer with dense curly fleece, narrow face, and cloven hooves",
        "A gentle flock grazer with thick cream fleece, narrow face, dark legs, and cloven hooves",
        "A compact pasture follower with curly wool, sideways ears, tapered muzzle, and small hooves",
        "A woolly hillside flock member with curled fleece, bare face, and cloven hooves",
        "A calm farmyard grazer with bulky curly fleece, delicate face, hooves, and shepherd nearby",
    ],
    "deer": [
        "A tan woodland browser with branching antlers, white rump, small hooves, and delicate legs",
        "A graceful forest visitor with antlered crown, upright ears, and pale tail underside",
        "A cautious meadow grazer with spotted coat, narrow face, fine legs, and antlers",
        "A light-footed woodland creature with branching antlers, white rump, and raised short tail",
        "A slim clearing dweller with watchful ears, small hooves, tan coat, and antlered head",
    ],
    "elephant": [
        "An enormous wrinkled gray giant with flexible trunk, tusks, fan-shaped ears, and pillar legs",
        "A towering savanna traveler with ivory tusks, swinging trunk, broad ears, and thick feet",
        "A massive gray wanderer with muscular trunk, spreading ears, domed head, and tusks",
        "A heavy grassland giant with curved tusks, long trunk, wrinkled hide, and columnar legs",
        "A huge dusty traveler with broad ears, tiny eyes, ivory tusks, and remarkable trunk",
    ],
}


# Revision v8 changes only candidates 10--15 in the three bear facets whose v7
# pool could not yield ten visually valid descriptions.  The v7 rows and images
# are archived before applying this revision; already valid candidates 1--9 are
# never regenerated.  Removing "standing upright" and the metaphorical "giant"
# reduces the recurring dog/kangaroo/humanoid morphology while retaining a
# name-free, visual description.
FACET_SUBJECT_OVERRIDES = {
    ("bear", "behavior_habits"): [
        "A huge shaggy brown animal with round ears, broad muzzle, shoulder hump, and curved claws",
        "A massive brown animal with dense fur, small round ears, powerful forelegs, and broad snout",
    ],
    ("bear", "indirect_holistic"): [
        "A huge shaggy brown animal with round ears, broad muzzle, shoulder hump, and curved claws",
        "A massive brown animal with dense fur, small round ears, powerful forelegs, and broad snout",
    ],
    ("bear", "movement"): [
        "A huge shaggy brown animal with round ears, broad muzzle, shoulder hump, and curved claws",
        "A massive brown animal with dense fur, small round ears, powerful forelegs, and broad snout",
    ],
}


FULL_DESCRIPTION_OVERRIDES = {
    ("bear", "indirect_holistic"): [
        "A huge shaggy brown animal with round ears, broad muzzle, shoulder hump, and curved claws anchors a story through posture.",
        "A hibernating brown cave giant with round ears, broad muzzle, short tail, and clawed paws rests among scattered berries.",
        "A massive hibernating brown animal emerges from a cave with round ears, shoulder hump, and broad clawed paws.",
        "A massive hibernating woodland giant with round ears, broad muzzle, short tail, and clawed paws sits beside honey jars.",
        "A massive brown animal with dense fur, small round ears, powerful forelegs, and broad snout forms one silhouette among objects.",
        "A hibernating shaggy cave giant with small round ears, wide snout, short tail, and heavy clawed feet overlooks a stream.",
    ],
    ("bear", "movement"): [
        "A huge shaggy brown animal charges through a shallow stream while swiping at leaping salmon with curved claws.",
        "A huge shaggy brown animal with round ears and curved claws runs beside a rushing river after leaping salmon.",
        "A thick-furred brown animal rises briefly, then drops onto four heavy paws and strides downhill.",
        "A massive thick-furred brown animal with round ears and broad paws lumbers across a rocky stream.",
        "A huge brown animal with dense shaggy fur, rounded ears, and long claws climbs over a fallen pine.",
        "A broad-shouldered shaggy brown animal runs downhill from a cave on four enormous clawed paws.",
    ],
}


def build_rows() -> list[dict]:
    rows = []
    for concept, subjects in CONCEPTS.items():
        for facet_id, predicates in FACETS.items():
            override_subjects = FACET_SUBJECT_OVERRIDES.get((concept, facet_id))
            facet_subjects = subjects[:3] + override_subjects if override_subjects else subjects
            full_overrides = FULL_DESCRIPTION_OVERRIDES.get((concept, facet_id))
            index = 0
            for subject in facet_subjects:
                for predicate in predicates:
                    index += 1
                    description = f"{subject} {predicate}."
                    source = (
                        "deterministic_balanced_grammar_v8_targeted_revision"
                        if override_subjects and index >= 10
                        else "deterministic_balanced_grammar_v7"
                    )
                    if full_overrides and index >= 10:
                        description = full_overrides[index - 10]
                        source = (
                            "hand_authored_v11_bear_indirect_revision"
                            if facet_id == "indirect_holistic"
                            else "hand_authored_v10_bear_movement_revision"
                        )
                    rows.append({
                        "candidate_id": f"{concept}_{facet_id}_{index:02d}",
                        "concept": concept,
                        "facet_id": facet_id,
                        "candidate_index": index,
                        "description": description,
                        "source": source,
                        "generation_metadata": {
                            "subject_variant": facet_subjects.index(subject) + 1,
                            "predicate_variant": predicates.index(predicate) + 1,
                        },
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concept", action="append")
    parser.add_argument("--facet")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    rows = build_rows()
    if args.concept:
        wanted = set(args.concept)
        rows = [row for row in rows if row["concept"] in wanted]
    if args.facet:
        rows = [row for row in rows if row["facet_id"] == args.facet]
    if not rows:
        raise ValueError("The requested concept/facet filter produced no candidates")
    if not args.concept and not args.facet:
        assert len(rows) == 1500
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
