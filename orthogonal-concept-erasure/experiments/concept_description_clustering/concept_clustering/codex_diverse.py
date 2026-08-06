from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from .lexical_baselines import DEFAULT_VECTORIZERS, _vectorizer
from .text_validation import opening_signature, token_jaccard, token_ngrams, words
from .utils import read_csv, read_jsonl, set_reproducible_seed, write_csv, write_jsonl


SUBJECTS = {
    "cat": [
        "whiskered house-prowler", "pointed-eared lap-companion", "slit-eyed indoor-hunter",
        "purring room-explorer", "retractable-clawed climber", "padded-paw windowsill-watcher",
        "long-tailed household-companion", "striped hearthside-prowler",
        "velvet-footed small-hunter", "curled-tail home-companion",
    ],
    "dog": [
        "wagging household-companion", "collar-wearing yard-guardian", "wet-nosed trail-helper",
        "leash-following walking-partner", "panting ball-chaser", "broad-pawed gate-watcher",
        "floppy-eared home-guardian", "keen-nosed scent-follower", "playful backyard-digger",
        "tongue-out family-companion",
    ],
    "fox": [
        "red-coated woodland-hunter", "white-tipped brush-bearer", "narrow-muzzled dusk-prowler",
        "black-legged forest-shadow", "sharp-eared field-stalker", "russet snowline-hunter",
        "slender nighttime-forager", "bushy-tailed den-visitor", "orange-chested hedgerow-stalker",
        "amber-eyed meadow-prowler",
    ],
    "bear": [
        "massive shaggy-forager", "round-eared timberland-giant", "broad-muzzled river-hunter",
        "heavy-pawed backcountry-wanderer", "thick-furred cave-sleeper", "powerful berry-gatherer",
        "towering honey-seeker", "short-tailed mountain-roamer", "bulky streamside-fisher",
        "long-clawed wilderness-resident",
    ],
}


FACET_VERBS = {
    "visual_appearance": ["displays", "presents", "reveals", "shows", "carries"],
    "body_parts": ["extends", "positions", "uses", "turns", "raises"],
    "movement": ["crosses", "approaches", "traverses", "circles", "climbs past"],
    "behavior_habits": ["investigates", "settles beside", "returns toward", "examines", "waits beside"],
    "sounds": ["makes", "releases", "produces", "answers with", "breaks silence using"],
    "habitat_scene": ["occupies", "moves through", "rests within", "emerges across", "forages around"],
    "human_interaction": ["approaches", "responds toward", "stays beside", "follows", "accepts attention from"],
    "function_ability": ["demonstrates", "performs", "uses", "shows", "relies upon"],
    "object_relationship": ["interacts with", "positions itself near", "reaches toward", "moves around", "rests against"],
    "indirect_holistic": ["creates", "forms", "becomes", "embodies", "completes"],
}


CUES = {
    "cat": {
        "visual_appearance": [
            "a compact coat patterned by soft stripes", "a low silhouette with alert triangular ears",
            "smooth fur surrounding reflective slit-shaped eyes", "a flexible tail curved above padded feet",
            "a neat face framed by long pale whiskers",
        ],
        "body_parts": [
            "long whiskers beside a small nose", "retractable claws above silent toe pads",
            "pointed ears that pivot toward faint motion", "a balancing tail behind flexible hind legs",
            "slit pupils within bright forward-facing eyes",
        ],
        "movement": [
            "a narrow ledge using nearly silent steps", "the room with a low stalking posture",
            "a fence through one fluid upward leap", "the floor before springing onto furniture",
            "a windowsill while its tail corrects balance",
        ],
        "behavior_habits": [
            "a warm patch before curling tightly", "moving string with focused paw taps",
            "a cardboard surface by repeated scratching", "dusty corners during a patient indoor patrol",
            "its face and forelegs with careful grooming strokes",
        ],
        "sounds": [
            "a steady vibration during relaxed contact", "a short questioning call beside an empty bowl",
            "a sharp warning hiss with flattened ears", "soft chirping notes while watching distant birds",
            "a rising trill during a familiar greeting",
        ],
        "habitat_scene": [
            "a sunny windowsill above indoor plants", "a quiet kitchen beside ceramic dishes",
            "a cushioned chair near warm afternoon light", "a garden wall overlooking moving leaves",
            "a dim hallway containing scattered household shadows",
        ],
        "human_interaction": [
            "an offered hand before rubbing its cheek", "a seated person by climbing onto their lap",
            "a familiar voice with an upright tail", "gentle brushing while remaining loosely curled",
            "nearby footsteps before following between rooms",
        ],
        "function_ability": [
            "precise balance along a narrow wooden edge", "rapid low-light tracking of tiny movement",
            "a sudden vertical jump from complete stillness", "quiet control of small household pests",
            "flexible landing posture after descending from furniture",
        ],
        "object_relationship": [
            "a loose yarn strand using alternating paws", "a scratching post beside upholstered furniture",
            "a cardboard box barely wider than its body", "curtain folds that conceal a moving silhouette",
            "a ceramic bowl before settling nearby",
        ],
        "indirect_holistic": [
            "the familiar outline guarding a nighttime windowsill", "a quiet domestic presence between sudden playful bursts",
            "the small shadow that chooses the warmest seat", "an alert room companion equally suited to stillness and pursuit",
            "the household hunter whose comfort arrives as vibration",
        ],
    },
    "dog": {
        "visual_appearance": [
            "an open-mouthed face above a moving tongue", "a sturdy coat ending in an energetic tail",
            "a broad chest beneath alert or gently folded ears", "a wet dark nose centered over a friendly muzzle",
            "four strong legs supporting a collar-ready frame",
        ],
        "body_parts": [
            "a sensitive nose toward traces on the ground", "a wagging tail behind broad hindquarters",
            "floppy ears that lift toward a familiar call", "a panting tongue between visible teeth",
            "padded paws beneath sturdy running legs",
        ],
        "movement": [
            "the yard in an eager bounding gait", "a footpath while pulling lightly against a leash",
            "open grass before turning sharply after a ball", "a doorway with rapid tail movement",
            "rough ground by following a scent trail",
        ],
        "behavior_habits": [
            "a buried object with repeated digging", "the gate during an attentive watch",
            "a thrown ball before carrying it back", "the floor beside a familiar person's chair",
            "new surroundings through persistent sniffing",
        ],
        "sounds": [
            "a resonant warning from behind the gate", "short excited calls during play",
            "a low rumble when unfamiliar movement approaches", "a drawn-out lonely call across the yard",
            "quick breathy whines beside a closed door",
        ],
        "habitat_scene": [
            "a fenced backyard containing scattered toys", "a neighborhood path beside a handled leash",
            "a family room near a cushioned floor bed", "a farm entrance requiring an alert watcher",
            "an open field suited to running and retrieval",
        ],
        "human_interaction": [
            "a returning person with vigorous tail sweeps", "a handler's gesture before sitting attentively",
            "children at play while carrying a soft ball", "a walking partner by matching their pace",
            "an offered palm through careful sniffing",
        ],
        "function_ability": [
            "long-distance scent tracking across broken ground", "retrieval of a thrown object on command",
            "an alert warning when strangers approach", "steady guidance beside a moving person",
            "coordinated herding pressure around livestock",
        ],
        "object_relationship": [
            "a worn tennis ball held between its jaws", "a leash extending toward a walking person",
            "a food bowl before waiting for permission", "a gate that defines its watching position",
            "a buried toy uncovered beneath loose soil",
        ],
        "indirect_holistic": [
            "the welcoming figure waiting behind a household door", "an energetic partnership built around voice and gesture",
            "the familiar shadow accompanying a neighborhood walk", "a social guardian balancing play with watchfulness",
            "the home companion whose excitement moves its whole rear body",
        ],
    },
    "fox": {
        "visual_appearance": [
            "rust-colored fur above dark lower legs", "a slender outline ending in a pale-tipped brush",
            "a narrow face beneath tall triangular ears", "an orange-red coat broken by a white chest",
            "a low graceful frame with an unusually full tail",
        ],
        "body_parts": [
            "a white tail tip beyond a dense brush", "black lower legs beneath a narrow torso",
            "large pointed ears above an elongated muzzle", "fine whiskers surrounding a dark nose",
            "light paws beneath lean springing limbs",
        ],
        "movement": [
            "snow before diving headfirst toward hidden movement", "a meadow through quick elastic bounds",
            "woodland cover using a low cautious trot", "a hedgerow before vanishing through a narrow gap",
            "the field with its full tail held level",
        ],
        "behavior_habits": [
            "faint ground sounds with tilted ears", "a concealed den near the woodland edge",
            "small prey through a sudden vertical pounce", "open ground during quiet twilight patrols",
            "spare food beneath loose leaves and soil",
        ],
        "sounds": [
            "a brief high call across evening woodland", "sharp chattering notes near a hidden den",
            "a thin scream carrying through darkness", "quiet yapping sounds during distant contact",
            "short warning barks from dense cover",
        ],
        "habitat_scene": [
            "a woodland margin beside open grass", "a snowy field containing faint prey tracks",
            "a brush-covered slope near a concealed den", "a moonlit hedgerow between farms and trees",
            "a quiet meadow during early dawn",
        ],
        "human_interaction": [
            "a distant observer before retreating into brush", "the edge of a garden while avoiding nearby footsteps",
            "a roadside light with a brief cautious pause", "discarded food while remaining ready to flee",
            "a camera trap during a solitary nighttime crossing",
        ],
        "function_ability": [
            "precise detection of prey beneath deep snow", "a sudden pounce guided almost entirely by sound",
            "silent travel between woodland and field cover", "rapid direction changes during close pursuit",
            "careful food caching beneath natural debris",
        ],
        "object_relationship": [
            "fallen leaves covering a recently hidden meal", "snow that conceals movement beneath the surface",
            "a den entrance screened by roots and brush", "field grass brushing against dark lower legs",
            "a pale moon outlining the full tail",
        ],
        "indirect_holistic": [
            "the red dusk shadow passing between field and forest", "a cautious hunter marked by a pale brush tip",
            "the solitary outline listening before a sudden leap", "an elusive edge-dweller adapted to cover and open ground",
            "the twilight figure that disappears after one sharp call",
        ],
    },
    "bear": {
        "visual_appearance": [
            "a towering shaggy frame above heavy paws", "dense brown fur around a broad pale muzzle",
            "a bulky silhouette with small rounded ears", "massive shoulders rising over a short tail",
            "a thick coat covering powerful curved limbs",
        ],
        "body_parts": [
            "long claws extending from broad front paws", "small round ears above a wide muzzle",
            "a short tail behind a massive torso", "powerful shoulders supporting a heavy head",
            "a sensitive nose ahead of dense facial fur",
        ],
        "movement": [
            "the riverbank with a rolling heavy gait", "a steep slope despite its bulky frame",
            "shallow water before striking toward fish", "forest ground on broad weight-bearing paws",
            "a fallen trunk through surprising climbing strength",
        ],
        "behavior_habits": [
            "berry bushes during long seasonal feeding", "a sheltered cave before winter sleep",
            "loose logs while searching for insects", "river shallows during a patient fishing period",
            "tree bark using its back and shoulders",
        ],
        "sounds": [
            "a deep warning rumble in dense woodland", "forceful huffs when danger comes closer",
            "low grunts while searching along the ground", "a loud roar across an open riverbank",
            "soft bawling notes from a smaller youngster",
        ],
        "habitat_scene": [
            "a mountain forest containing berry thickets", "a cold river crowded by migrating fish",
            "a rocky cave beneath winter snow", "an alpine clearing surrounded by fallen logs",
            "a dense woodland trail marked by broad tracks",
        ],
        "human_interaction": [
            "a distant hiker before turning toward cover", "secured food containers near a wilderness campsite",
            "a ranger's observation from across the river", "a roadside vehicle while remaining beyond the barrier",
            "an empty trail after detecting approaching voices",
        ],
        "function_ability": [
            "powerful fishing strikes in fast shallow water", "long seasonal fasting inside a sheltered den",
            "strong digging through roots and compact soil", "sensitive scent detection across a wide forest area",
            "surprising climbing force despite substantial body weight",
        ],
        "object_relationship": [
            "a fallen log opened by long curved claws", "river stones surrounding a fishing position",
            "a tree trunk used for forceful scratching", "berry branches bent beneath broad paws",
            "a sealed wilderness container investigated by scent",
        ],
        "indirect_holistic": [
            "the huge forest presence that vanishes before winter", "a riverbank giant waiting for silver movement",
            "the shaggy mountain silhouette built for strength and fasting", "a solitary forager equally comfortable digging and climbing",
            "the broad-tracked resident of cave river and berry slope",
        ],
    },
}


SCENE_ADJECTIVES = [
    "amber", "misty", "dappled", "quiet", "silver", "windy", "shaded", "sunlit", "dusky", "frosted",
    "rainy", "mossy", "golden", "cool", "leafy", "muted", "cloudy", "still", "pale", "soft",
    "bright", "dim", "wooden", "grassy", "rocky", "warm", "shadowed", "open", "narrow", "remote",
    "copper", "blue", "drifting", "crisp", "verdant", "russet", "tranquil", "breezy", "speckled", "clear",
]
SCENE_NOUNS = [
    "doorway", "clearing", "path", "window", "garden", "slope", "riverbank", "meadow", "hallway", "fence",
    "trail", "porch", "grove", "courtyard", "kitchen", "field", "stream", "room", "hedgerow", "ledge",
    "campsite", "thicket", "farmyard", "snowbank", "woodland", "terrace", "bridge", "fireside", "gate", "hillside",
    "orchard", "balcony", "ravine", "shoreline", "workshop", "lane", "foothill", "marsh", "barn", "plaza",
    "alcove", "ridge", "veranda", "crossing", "pasture", "brook", "hedge", "stairway", "cabin", "bank",
    "yard", "trailhead", "copse", "pond", "deck", "valley", "corner", "hedgerow", "outcrop", "glade",
]
ADVERBS = [
    "carefully", "steadily", "quietly", "attentively", "deliberately", "gracefully", "patiently", "curiously",
    "cautiously", "confidently", "gently", "silently", "alertly", "methodically", "restlessly", "calmly",
    "swiftly", "warily", "precisely", "eagerly", "slowly", "smoothly", "vigilantly", "softly", "purposefully",
    "observantly", "measuredly", "lightly", "firmly", "readily", "watchfully", "intently", "leisurely",
    "tentatively", "rhythmically", "nimbly", "solidly", "knowingly", "comfortably", "inquisitively",
    "restrainedly", "energetically", "discreetly", "persistently", "prudently", "steadfastly", "acutely",
    "responsively", "unhurriedly", "dexterously",
]
SUBJECT_MODIFIERS = [
    "alert", "watchful", "quiet", "nimble", "restless", "patient", "curious", "wary", "steady",
    "graceful", "lively", "calm", "intent", "cautious", "observant", "swift", "focused", "gentle",
    "solitary", "active", "vigilant", "measured", "eager", "poised", "attentive",
]

PAIR_CONNECTORS = {
    "visual_appearance": ["beside", "with", "alongside", "beneath", "around"],
    "body_parts": ["beside", "above", "alongside", "near", "around"],
    "movement": ["with", "using", "through", "over", "along"],
    "behavior_habits": ["during", "through", "beside", "around", "after"],
    "sounds": ["during", "around", "beside", "through", "under"],
    "habitat_scene": ["near", "among", "beside", "within", "under"],
    "human_interaction": ["before", "beside", "during", "toward", "around"],
    "function_ability": ["along", "through", "across", "during", "around"],
    "object_relationship": ["beside", "against", "around", "beneath", "near"],
    "indirect_holistic": ["within", "beside", "across", "under", "around"],
}
PAIR_MODIFIERS = [
    "distinct", "visible", "prominent", "defined", "clear", "subtle", "bright", "muted", "soft", "sharp",
    "delicate", "strong", "familiar", "nearby", "natural", "quiet", "active", "steady", "graceful", "compact",
    "textured", "balanced", "recognizable", "focused", "striking",
]


def _edge_pair(phrase: str) -> tuple[str, str]:
    tokens = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", phrase)
    relation_words = {
        "above", "across", "after", "against", "ahead", "along", "around", "before",
        "behind", "beneath", "beside", "between", "beyond", "by", "containing", "covering",
        "curved", "during", "ending", "equally", "extending", "framed", "from", "held",
        "inside", "into", "marked", "near", "of", "on", "over", "patterned", "rising",
        "screened", "surrounding", "supporting", "that", "through", "toward", "under",
        "uncovered", "using", "while", "whose", "with", "within",
    }
    head_end = len(tokens)
    for index, token in enumerate(tokens):
        if index >= 2 and token.casefold() in relation_words:
            head_end = index
            break
    left = tokens[:head_end]
    right = tokens[-2:]
    while len(right) > 1 and right[0].casefold() in {
        "a", "an", "and", "at", "by", "for", "from", "in", "into", "its", "near", "of", "on",
        "over", "the", "their", "through", "to", "toward", "under", "with", "within",
    }:
        right = right[1:]
    return " ".join(left), " ".join(right)


def _paired_cue(concept: str, facet: str, combo: int) -> str:
    cue_index = combo % 25
    left_index = cue_index // 5
    right_index = cue_index % 5
    left, _ = _edge_pair(CUES[concept][facet][left_index])
    _, right = _edge_pair(CUES[concept][facet][right_index])
    connector = PAIR_CONNECTORS[facet][(left_index + right_index) % 5]
    left_tokens = left.split()
    group_offset = list(CUES).index(concept) * 7 + list(CUES[concept]).index(facet) * 3
    left_modifier = PAIR_MODIFIERS[(cue_index + group_offset) % len(PAIR_MODIFIERS)]
    if left_tokens[0].casefold() in {"a", "an", "the"}:
        determiner = ["a", "the", "one", "this", "that"][left_index]
        left_tokens[0] = determiner
        if len(left_tokens) <= 3:
            left_tokens.insert(1, left_modifier)
        else:
            left_tokens[1] = left_modifier
        if left_tokens[0].casefold() in {"a", "an"}:
            left_tokens[0] = "an" if left_tokens[1][0].casefold() in "aeiou" else "a"
    elif len(left_tokens) <= 2:
        left_tokens.insert(0, left_modifier)
    else:
        left_tokens[0] = left_modifier
    right_modifier = PAIR_MODIFIERS[(cue_index + group_offset + 7) % len(PAIR_MODIFIERS)]
    return f"{' '.join(left_tokens)} {connector} {right_modifier} {right}"


def _render(pattern: int, subject: str, adverb: str, verb: str, cue: str, scene: str) -> str:
    verb_tokens = verb.split()
    if len(verb_tokens) == 1:
        predicate = f"{adverb} {verb}"
    else:
        predicate = " ".join([verb_tokens[0], adverb, *verb_tokens[1:]])
    article = "An" if subject[0].casefold() in "aeiou" else "A"
    clause = f"{article} {subject} {predicate} {cue}"
    variants = [
        f"{clause} near {scene}.",
        f"{clause} beside {scene}.",
        f"{clause} amid {scene}.",
        f"{clause} beyond {scene}.",
        f"{clause} around {scene}.",
        f"{clause} against {scene}.",
        f"{clause} along {scene}.",
        f"{clause} inside {scene}.",
        f"{clause} past {scene}.",
        f"{clause} across {scene}.",
        f"{clause} under {scene}.",
        f"{clause} before {scene}.",
    ]
    return variants[pattern % len(variants)]


def build_codex_diverse_candidates(config: dict[str, Any], output_path: str | Path) -> list[dict[str, Any]]:
    """Build a deterministic single-source pool, honestly labeled codex_diverse."""
    generator_seed = int(config.get("dataset_design", {}).get("generator_seed", 8137))
    variation = (generator_seed - 8137) % 250
    set_reproducible_seed(generator_seed)
    concepts = [item["name"] for item in config["concepts"]]
    unsupported = sorted(set(concepts) - set(SUBJECTS))
    if unsupported:
        raise ValueError(f"codex_diverse profiles are not defined for: {unsupported}")
    facets = [item["id"] for item in config["facets"]]
    per_group = int(config["candidate_validation"]["candidates_per_concept_facet"])
    minimum_diverse = min(
        per_group,
        int(config.get("dataset_design", {}).get("minimum_diverse_per_group", min(12, per_group))),
    )
    if per_group > 25:
        raise ValueError("The deterministic codex_diverse builder currently supports at most 25 per group")

    rng = np.random.default_rng(generator_seed)
    rows: list[dict[str, Any]] = []
    seen_trigrams: set[tuple[str, ...]] = set()
    seen_openings: set[str] = set()
    previous_texts: list[str] = []
    global_index = 0
    concept_accepted_counts: Counter[str] = Counter()
    used_scenes: set[tuple[str, str]] = set()
    used_scene_adverbs: set[tuple[str, str]] = set()
    for concept in concepts:
        for facet in facets:
            accepted_in_group = 0
            attempts = 0
            rejection_counts: Counter[str] = Counter()
            first_collision: tuple[str, ...] | None = None
            while accepted_in_group < per_group and attempts < 10000:
                attempts += 1
                candidate_index = accepted_in_group + 1
                combo = attempts - 1
                concept_index = concept_accepted_counts[concept]
                subject_core = SUBJECTS[concept][(concept_index + variation) % len(SUBJECTS[concept])]
                subject_modifier = SUBJECT_MODIFIERS[
                    (concept_index // len(SUBJECTS[concept]) + concepts.index(concept) * 7 + variation)
                    % len(SUBJECT_MODIFIERS)
                ]
                if bool(config.get("dataset_design", {}).get("force_full_body_anchor", False)):
                    subject_parts = subject_core.split()
                    subject = " ".join([subject_parts[0], subject_modifier, "full-bodied", *subject_parts[1:]])
                else:
                    subject = f"{subject_modifier} {subject_core}"
                verb = FACET_VERBS[facet][combo % len(FACET_VERBS[facet])]
                cue = _paired_cue(concept, facet, combo + variation)
                subject_cycle, subject_slot = divmod(concept_index, len(SUBJECTS[concept]))
                concept_offset = concepts.index(concept) * 7
                adverb = ADVERBS[(subject_slot * 5 + subject_cycle + concept_offset + variation) % len(ADVERBS)]
                scene_index = (global_index * 37 + combo + variation * 11) % (len(SCENE_ADJECTIVES) * len(SCENE_NOUNS))
                scene_adjective = SCENE_ADJECTIVES[scene_index // len(SCENE_NOUNS)]
                scene_noun = SCENE_NOUNS[scene_index % len(SCENE_NOUNS)]
                if (scene_adjective, scene_noun) in used_scenes or (scene_noun, adverb) in used_scene_adverbs:
                    rejection_counts["scene_reuse"] += 1
                    continue
                scene = f"{scene_adjective} {scene_noun}"
                pattern = (global_index + combo * 5 + variation) % 12
                description = _render(pattern, subject, adverb, verb, cue, scene)
                token_count = len(words(description))
                if not 8 <= token_count <= 20:
                    rejection_counts["length"] += 1
                    continue
                trigrams = token_ngrams(description, 3)
                opening = opening_signature(description, 3)
                collisions = [gram for gram in trigrams if gram in seen_trigrams]
                if collisions:
                    rejection_counts["trigram"] += 1
                    first_collision = first_collision or collisions[0]
                    continue
                if opening in seen_openings:
                    rejection_counts["opening"] += 1
                    continue
                if any(token_jaccard(description, earlier) >= 0.72 for earlier in previous_texts):
                    rejection_counts["near_duplicate"] += 1
                    continue
                row = {
                    "candidate_id": f"codex_diverse__{concept}__{facet}__{candidate_index:02d}",
                    "concept": concept,
                    "facet_id": facet,
                    "candidate_index": candidate_index,
                    "description": description,
                    "source": "codex_diverse",
                    "generation_metadata": {
                        "generator": "Codex deterministic diverse corpus builder",
                        "generator_seed": generator_seed,
                        "syntax_family": f"render_{pattern:02d}",
                        "single_source": True,
                    },
                }
                rows.append(row)
                previous_texts.append(description)
                seen_trigrams.update(trigrams)
                seen_openings.add(opening)
                used_scenes.add((scene_adjective, scene_noun))
                used_scene_adverbs.add((scene_noun, adverb))
                accepted_in_group += 1
                global_index += 1
                concept_accepted_counts[concept] += 1
            if accepted_in_group < minimum_diverse:
                raise RuntimeError(
                    f"Could only build {accepted_in_group}/{per_group} diverse candidates for {concept}/{facet}; "
                    f"rejections={dict(rejection_counts)} first_collision={first_collision}"
                )
            # Preserve the requested initial pool size even when the strict global
            # trigram/opening constraints exhaust a group.  These fallback rows
            # are still passed through the normal validator and are expected to
            # be rejected with explicit reasons; they are never silently used.
            while accepted_in_group < per_group:
                candidate_index = accepted_in_group + 1
                fallback_combo = candidate_index - 1
                description = _render(
                    fallback_combo,
                    (
                        " ".join([
                            SUBJECTS[concept][(fallback_combo + variation) % len(SUBJECTS[concept])].split()[0],
                            SUBJECT_MODIFIERS[(fallback_combo + variation) % len(SUBJECT_MODIFIERS)],
                            "full-bodied",
                            *SUBJECTS[concept][(fallback_combo + variation) % len(SUBJECTS[concept])].split()[1:],
                        ])
                        if bool(config.get("dataset_design", {}).get("force_full_body_anchor", False))
                        else f"{SUBJECT_MODIFIERS[(fallback_combo + variation) % len(SUBJECT_MODIFIERS)]} "
                        f"{SUBJECTS[concept][(fallback_combo + variation) % len(SUBJECTS[concept])]}"
                    ),
                    ADVERBS[fallback_combo % len(ADVERBS)],
                    FACET_VERBS[facet][fallback_combo % len(FACET_VERBS[facet])],
                    _paired_cue(concept, facet, fallback_combo + variation),
                    f"{SCENE_ADJECTIVES[(fallback_combo + variation) % len(SCENE_ADJECTIVES)]} "
                    f"{SCENE_NOUNS[(fallback_combo + variation) % len(SCENE_NOUNS)]}",
                )
                rows.append({
                    "candidate_id": f"codex_diverse__{concept}__{facet}__{candidate_index:02d}",
                    "concept": concept,
                    "facet_id": facet,
                    "candidate_index": candidate_index,
                    "description": description,
                    "source": "codex_diverse",
                    "generation_metadata": {
                        "generator": "Codex deterministic diverse corpus builder",
                        "generator_seed": generator_seed,
                        "syntax_family": f"render_{fallback_combo % 12:02d}",
                        "single_source": True,
                        "fallback_expected_text_rejection": True,
                    },
                })
                accepted_in_group += 1
    write_jsonl(output_path, rows, overwrite=True)
    return rows


def score_and_select_tfidf_hard(
    config: dict[str, Any],
    validation_dir: str | Path,
    selected_path: str | Path,
    scores_path: str | Path,
) -> list[dict[str, Any]]:
    """Use strictly out-of-fold TF-IDF probabilities to rank candidates."""
    validation_dir = Path(validation_dir)
    candidates = read_jsonl(validation_dir / "candidate_descriptions.jsonl")
    valid = {
        row["candidate_id"]
        for row in read_csv(validation_dir / "candidate_text_validation.csv")
        if row.get("candidate_id") not in {"__group_count__", "__source_count__"}
        and str(row.get("text_valid", "")).casefold() == "true"
    }
    candidates = [row for row in candidates if row["candidate_id"] in valid]
    concepts = [item["name"] for item in config["concepts"]]
    labels = np.array([concepts.index(row["concept"]) for row in candidates])
    texts = [row["description"] for row in candidates]
    folds = int(config.get("tfidf_hard_selection", {}).get("cv_folds", 5))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=9127)
    specs = config.get("lexical_baselines", {}).get("vectorizers", DEFAULT_VECTORIZERS)
    by_candidate: dict[str, dict[str, Any]] = {
        row["candidate_id"]: {
            "candidate_id": row["candidate_id"], "concept": row["concept"],
            "facet_id": row["facet_id"], "description": row["description"],
        }
        for row in candidates
    }
    summary_rows = []
    for spec in specs:
        probabilities = np.zeros((len(candidates), len(concepts)), dtype=np.float64)
        for train, test in cv.split(texts, labels):
            pipeline = make_pipeline(
                _vectorizer(spec),
                LogisticRegression(max_iter=5000, random_state=0, solver="liblinear"),
            )
            pipeline.fit([texts[index] for index in train], labels[train])
            fold_probabilities = pipeline.predict_proba([texts[index] for index in test])
            model_classes = pipeline[-1].classes_.astype(int)
            probabilities[np.ix_(test, model_classes)] = fold_probabilities
        predicted = probabilities.argmax(axis=1)
        accuracy = float(np.mean(predicted == labels))
        summary_rows.append({
            "representation": spec["name"], "oof_accuracy": accuracy,
            "cv_folds": folds, "cv_seed": 9127, "n_candidates": len(candidates),
        })
        for index, candidate in enumerate(candidates):
            target_id = labels[index]
            target_score = float(probabilities[index, target_id])
            order = np.argsort(-probabilities[index])
            best_other = max(float(probabilities[index, other]) for other in range(len(concepts)) if other != target_id)
            prefix = str(spec["name"])
            record = by_candidate[candidate["candidate_id"]]
            record[f"{prefix}__predicted_class"] = concepts[int(predicted[index])]
            record[f"{prefix}__target_probability"] = target_score
            record[f"{prefix}__target_rank"] = int(np.where(order == target_id)[0][0]) + 1
            record[f"{prefix}__target_margin"] = target_score - best_other

    score_rows = []
    for candidate in candidates:
        record = by_candidate[candidate["candidate_id"]]
        prefixes = [str(spec["name"]) for spec in specs]
        incorrect = sum(record[f"{prefix}__predicted_class"] != candidate["concept"] for prefix in prefixes)
        probabilities = [float(record[f"{prefix}__target_probability"]) for prefix in prefixes]
        margins = [float(record[f"{prefix}__target_margin"]) for prefix in prefixes]
        record["tfidf_incorrect_model_count"] = incorrect
        record["tfidf_mean_target_probability"] = float(np.mean(probabilities))
        record["tfidf_mean_target_margin"] = float(np.mean(margins))
        record["tfidf_min_target_margin"] = float(np.min(margins))
        record["tfidf_difficulty_score"] = float(2.0 * incorrect + (1.0 - np.mean(probabilities)) - np.mean(margins))
        score_rows.append(record)

    per_group = int(config.get("tfidf_hard_selection", {}).get("generation_candidates_per_concept_facet", 18))
    selected_ids: set[str] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in score_rows:
        grouped[(record["concept"], record["facet_id"])].append(record)
    shortages = []
    for concept in concepts:
        for facet in [item["id"] for item in config["facets"]]:
            group = sorted(
                grouped[(concept, facet)],
                key=lambda row: (-row["tfidf_difficulty_score"], row["candidate_id"]),
            )
            if len(group) < per_group:
                shortages.append({"concept": concept, "facet_id": facet, "required": per_group, "available": len(group)})
            selected_ids.update(row["candidate_id"] for row in group[:per_group])
    if shortages:
        write_csv(Path(scores_path).with_name("tfidf_selection_shortages.csv"), shortages)
        raise RuntimeError(f"TF-IDF hard selection has {len(shortages)} deficient groups")

    candidate_map = {row["candidate_id"]: row for row in candidates}
    selected = []
    for record in sorted(score_rows, key=lambda row: row["candidate_id"]):
        record["selected_for_generation"] = record["candidate_id"] in selected_ids
        if record["selected_for_generation"]:
            candidate = dict(candidate_map[record["candidate_id"]])
            candidate["tfidf_hardness"] = {
                key: value for key, value in record.items()
                if key.startswith("tfidf_") or "__" in key
            }
            selected.append(candidate)
    write_csv(scores_path, score_rows)
    write_csv(Path(scores_path).with_name("tfidf_oof_summary.csv"), summary_rows)
    write_jsonl(selected_path, selected, overwrite=True)
    return selected
