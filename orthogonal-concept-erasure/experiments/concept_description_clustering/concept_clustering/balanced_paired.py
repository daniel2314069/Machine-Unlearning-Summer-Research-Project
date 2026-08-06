from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from transformers import CLIPTokenizer

from scripts.eot_spherical_clustering import (
    evaluate_after_clustering,
    extract_eot_embeddings,
    fit_spherical_kmeans,
    normalize_rows,
)

from .codex_diverse import CUES as LEGACY_CUES
from .codex_diverse import SUBJECTS as LEGACY_SUBJECTS
from .embeddings import _decoded_word, _extract_contextual, _normalize_tensor, _selected_token_audit
from .generation_validation import run_generation_validation
from .modeling import load_original_pipeline
from .utils import atomic_write_text, package_versions, read_csv, read_jsonl, write_csv, write_jsonl


PRIMARY_CONCEPTS = ["cat", "dog", "fox", "bear"]
EXTENSION_CONCEPTS = PRIMARY_CONCEPTS + ["wolf", "rabbit", "deer", "horse"]
FACETS = [
    "coat_color_texture",
    "head_face_ears",
    "limbs_paws_tail",
    "body_size_shape",
    "movement",
    "behavior_foraging",
    "habitat_scene",
    "interaction_with_objects_or_people",
    "interaction_with_other_animals",
    "sound_or_sensory_cues",
]
LENGTH_SCHEDULE = ["short", "short", "medium", "long", "long"]
LENGTH_RANGES = {"short": (14, 19), "medium": (21, 26), "long": (28, 33)}
FIXED_SUFFIX = " This sentence describes the concept"
REQUESTED_OUTPUTS = [
    "experiment_config.json", "shared_slots.csv", "accepted_descriptions.jsonl",
    "rejected_candidates.jsonl", "dataset_balance.csv", "dataset_validation.json",
    "fixed_suffix_embeddings.npy", "eot_embeddings.npy", "fixed_suffix_metrics.json",
    "eot_metrics.json", "metrics_comparison.csv", "confound_alignment.csv",
    "confusion_fixed_suffix.csv", "confusion_eot.csv", "confusion_fixed_suffix.png",
    "confusion_eot.png", "pca_fixed_suffix_true.png", "pca_fixed_suffix_predicted.png",
    "pca_eot_true.png", "pca_eot_predicted.png", "report.md",
]


FORBIDDEN_TERMS = {
    "cat": ["cat", "cats", "feline", "felines", "kitten", "kittens", "kitty", "kitties", "tomcat", "tomcats"],
    "dog": ["dog", "dogs", "canine", "canines", "puppy", "puppies", "hound", "hounds", "pooch", "pooches"],
    "fox": ["fox", "foxes", "vulpine", "vixen", "vixens"],
    "bear": ["bear", "bears", "ursine", "cub", "cubs"],
    "wolf": ["wolf", "wolves", "wolfish", "lupine"],
    "rabbit": ["rabbit", "rabbits", "bunny", "bunnies", "hare", "hares", "lapine"],
    "deer": ["deer", "cervid", "cervids", "doe", "does", "fawn", "fawns", "stag", "stags"],
    "horse": ["horse", "horses", "pony", "ponies", "equine", "stallion", "stallions", "mare", "mares", "foal", "foals"],
}


NEW_SUBJECTS = {
    "wolf": [
        "gray pack hunter", "long-legged forest tracker", "thick-coated wilderness runner",
        "amber-eyed pack sentinel", "broad-pawed northern hunter", "bushy-tailed woodland tracker",
        "deep-chested roaming predator", "sharp-eared snow traveler", "social mountain hunter",
        "pale-muzzled forest sentinel",
    ],
    "rabbit": [
        "long-eared meadow grazer", "soft-coated burrow dweller", "small hind-legged field browser",
        "white-tailed garden visitor", "round-eyed grassland nibbler", "quick-footed burrow resident",
        "compact meadow forager", "velvet-eared field grazer", "short-tailed garden browser",
        "twitch-nosed ground dweller",
    ],
    "deer": [
        "antlered woodland browser", "slender-legged forest grazer", "white-tailed meadow visitor",
        "large-eared woodland browser", "spotted forest youngster", "graceful clearing grazer",
        "split-hoofed trail visitor", "alert brown forest browser", "long-legged herd grazer",
        "branch-antlered woodland traveler",
    ],
    "horse": [
        "long-maned pasture runner", "broad-chested stable resident", "flowing-tailed field grazer",
        "single-hoofed riding companion", "powerful paddock runner", "tall long-faced grazer",
        "saddle-trained trail companion", "muscular farmyard traveler", "upright-eared pasture grazer",
        "swift-maned open-field runner",
    ],
}


SHORT_SUBJECTS = {
    "cat": [
        "whiskered lap companion", "purring house prowler", "retractable-clawed mouser",
        "slit-eyed room hunter", "padded-paw windowsill watcher", "pointed-eared hearth prowler",
        "long-whiskered home companion", "soft-pawed indoor hunter", "striped household mouser",
        "upright-tailed lap companion",
    ],
    "dog": [
        "leash-wearing companion", "wagging yard guardian", "wet-nosed trail helper",
        "panting ball retriever", "collared gate watcher", "floppy-eared home guardian",
        "keen-nosed scent tracker", "broad-pawed walking partner", "playful backyard digger",
        "tail-wagging family companion",
    ],
    "fox": [
        "red-coated dusk hunter", "white-tipped brush bearer", "narrow-muzzled night prowler",
        "black-legged forest hunter", "sharp-eared field stalker", "russet snowline hunter",
        "bushy-tailed den visitor", "orange-chested hedge stalker", "amber-eyed meadow prowler",
        "slender twilight forager",
    ],
    "bear": [
        "massive shaggy forager", "round-eared forest giant", "broad-muzzled river hunter",
        "heavy-pawed mountain roamer", "long-clawed cave sleeper", "powerful berry gatherer",
        "towering honey seeker", "bulky streamside fisher", "thick-furred woodland giant",
        "broad-pawed wilderness forager",
    ],
    "wolf": [
        "gray pack hunter", "amber-eyed pack sentinel", "broad-pawed northern hunter",
        "sharp-eared snow tracker", "deep-chested pack runner", "bushy-tailed forest tracker",
        "pale-muzzled pack sentinel", "long-legged wilderness hunter", "social mountain predator",
        "thick-coated northern tracker",
    ],
    "rabbit": [
        "long-eared meadow grazer", "twitch-nosed burrow dweller", "white-tailed garden browser",
        "hind-legged field nibbler", "velvet-eared grass grazer", "quick-footed burrow resident",
        "round-eyed clover browser", "soft-coated meadow nibbler", "short-tailed garden visitor",
        "long-eared ground forager",
    ],
    "deer": [
        "antlered woodland browser", "white-tailed meadow grazer", "split-hoofed forest visitor",
        "large-eared woodland grazer", "branch-antlered forest browser", "slender-legged herd grazer",
        "alert brown woodland browser", "long-legged clearing grazer", "pale-rumped forest visitor",
        "fine-hoofed meadow browser",
    ],
    "horse": [
        "long-maned pasture runner", "single-hoofed riding companion", "flowing-tailed field grazer",
        "saddle-trained trail companion", "broad-chested paddock runner", "tall long-faced grazer",
        "bridled stable companion", "muscular farmyard runner", "swift-maned field traveler",
        "upright-eared pasture grazer",
    ],
}

COMPACT_SHORT_SUBJECTS = {
    "wolf": [
        "wild gray pack hunter", "howling snow hunter", "forest pack sentinel", "broad-pawed pack tracker",
        "amber-eyed pack hunter", "northern pack runner", "bushy-tailed pack tracker", "mountain pack hunter",
        "pale-muzzled pack sentinel", "long-legged pack tracker",
    ],
    "rabbit": [
        "long-eared grazer", "burrow nibbler", "white-tailed browser", "meadow grazer",
        "twitch-nosed nibbler", "hind-legged browser", "clover grazer", "garden nibbler",
        "velvet-eared browser", "burrow resident",
    ],
    "deer": [
        "antlered browser", "woodland grazer", "white-tailed browser", "split-hoofed grazer",
        "forest herd grazer", "meadow browser", "branch-antlered visitor", "clearing grazer",
        "long-legged browser", "pale-rumped grazer",
    ],
    "horse": [
        "maned pasture runner", "saddle companion", "hoofed field runner", "stable grazer",
        "paddock runner", "bridled trail companion", "flowing-tailed grazer", "farm runner",
        "riding companion", "long-faced grazer",
    ],
}

STRONG_EXTENSION_SUBJECTS = {
    "wolf": [
        "large gray howling pack hunter", "wild gray pack predator", "broad-pawed northern pack tracker",
        "amber-eyed wilderness pack sentinel", "long-legged snow pack hunter", "deep-chested gray pack runner",
        "bushy-tailed mountain pack tracker", "pale-muzzled forest pack hunter", "thick-coated howling sentinel",
        "powerful wild pack traveler",
    ],
    "rabbit": [
        "small long-eared burrow grazer", "white-tailed hind-legged meadow browser", "twitch-nosed clover nibbler",
        "soft-coated long-eared burrow resident", "compact garden grazer with powerful hind legs",
        "round-eyed meadow nibbler with tall ears", "quick-footed white-tailed field browser",
        "velvet-eared ground grazer", "small hopping burrow forager", "long-eared grassland nibbler",
    ],
    "deer": [
        "antlered split-hoofed woodland browser", "white-tailed long-legged meadow grazer",
        "branch-antlered forest herd browser", "large-eared split-hoofed clearing grazer",
        "slender-legged antlered woodland traveler", "alert white-tailed forest browser",
        "graceful antlered meadow grazer", "fine-hoofed long-legged herd visitor",
        "brown woodland browser with branching antlers", "pale-rumped forest herd grazer",
    ],
    "horse": [
        "tall long-maned single-hoofed pasture runner", "saddle-wearing flowing-tailed riding companion",
        "broad-chested maned paddock grazer", "powerful single-hoofed stable runner",
        "long-faced bridled trail companion", "muscular maned farmyard traveler",
        "swift flowing-tailed field runner", "upright-eared single-hoofed pasture grazer",
        "tall saddle-trained riding companion", "long-maned stable grazer with powerful legs",
    ],
}


NEW_CUES: dict[str, dict[str, list[str]]] = {
    "wolf": {
        "coat_color_texture": [
            "a layered gray coat with pale guard hairs", "dense silver-brown fur suited to winter",
            "dark saddle markings across coarse back fur", "a thick charcoal coat around a lighter chest",
            "mottled gray hair blending into snowy brush",
        ],
        "head_face_ears": [
            "upright triangular ears above an elongated muzzle", "amber eyes set in a broad gray face",
            "a dark nose beyond a pale tapered muzzle", "alert ears framing a heavy-cheeked head",
            "a focused forward gaze beneath pointed ears",
        ],
        "limbs_paws_tail": [
            "broad paws beneath long endurance-built legs", "a low bushy tail behind powerful hindquarters",
            "large snow-spreading feet and straight forelegs", "dark padded feet supporting a steady trot",
            "a full trailing tail balancing lean limbs",
        ],
        "body_size_shape": [
            "a deep chest and lean long-backed frame", "a tall athletic silhouette heavier than a yard companion",
            "powerful shoulders narrowing toward the flanks", "a rangy body built for sustained travel",
            "a substantial northern frame with a tucked waist",
        ],
        "movement": [
            "an efficient ground-covering trot through snow", "coordinated running beside several pack members",
            "silent stepping between closely spaced pines", "a rapid pursuit over broken mountain ground",
            "measured circling around an unfamiliar scent",
        ],
        "behavior_foraging": [
            "cooperative pursuit of large hoofed prey", "careful scent inspection along a territorial boundary",
            "food sharing among an organized family group", "patient tracking across old snow prints",
            "a coordinated watch before the group advances",
        ],
        "habitat_scene": [
            "a snowy conifer forest crossed by pack tracks", "an open tundra ridge beneath cold wind",
            "a mountain valley bordered by dense woodland", "a remote pine trail at winter dusk",
            "a broad northern plain scattered with scrub",
        ],
        "interaction_with_objects_or_people": [
            "cautious inspection of a distant trail camera", "wide avoidance of boots near a remote campsite",
            "a brief pause beside an old boundary post", "careful sniffing around an abandoned sled track",
            "watchful distance from a ranger across the valley",
        ],
        "interaction_with_other_animals": [
            "close formation beside related pack members", "coordinated pressure around a fleeing hoofed grazer",
            "nose-to-muzzle greeting within the family group", "protective positioning near younger pack members",
            "a tense boundary display toward a rival group",
        ],
        "sound_or_sensory_cues": [
            "a long resonant howl answered across the valley", "exceptional scent tracking over frozen ground",
            "low social whines exchanged within the pack", "upright ears locating distant movement under snow",
            "a deep warning growl beneath raised neck fur",
        ],
    },
    "rabbit": {
        "coat_color_texture": [
            "soft brown fur mottled like dry grass", "a dense white winter coat with gray ear tips",
            "smooth tan hair above a pale belly", "velvety gray fur around a bright white tail",
            "a warm chestnut coat blending into meadow soil",
        ],
        "head_face_ears": [
            "very long upright ears above round side-set eyes", "a constantly twitching nose beneath fine whiskers",
            "broad listening ears framing a compact face", "large dark eyes and a split upper lip",
            "soft cheeks below tall independently turning ears",
        ],
        "limbs_paws_tail": [
            "oversized hind feet behind small front paws", "powerful folded rear legs beneath a round body",
            "a small bright tail above long jumping feet", "softly furred soles and compact forepaws",
            "long hind limbs prepared for sudden bounds",
        ],
        "body_size_shape": [
            "a compact rounded body held close to the ground", "a lightweight frame with prominent folded haunches",
            "a small pear-shaped silhouette and short neck", "a delicate torso supported by enlarged hindquarters",
            "a low oval profile ending in a tiny tail",
        ],
        "movement": [
            "rapid bounding toward a nearby burrow entrance", "a sudden zigzag escape across short grass",
            "small cautious hops followed by complete stillness", "an upright leap powered by long rear legs",
            "quiet crouching before a swift meadow dash",
        ],
        "behavior_foraging": [
            "steady nibbling of clover beside the burrow", "careful clipping of tender grass at dawn",
            "upright scanning between short feeding sessions", "chewing leafy stems while remaining near cover",
            "a shallow scrape exposing roots and fresh shoots",
        ],
        "habitat_scene": [
            "a grassy meadow dotted with hidden burrow openings", "a garden edge beside low leafy cover",
            "a sandy bank containing several narrow tunnels", "a quiet field beneath dense hedgerow shelter",
            "a woodland clearing covered in clover",
        ],
        "interaction_with_objects_or_people": [
            "gentle sniffing of a hand holding leafy greens", "quick retreat beneath a low garden bench",
            "curious inspection of a cardboard tunnel", "quiet resting beside a ceramic food dish",
            "soft pawing at the edge of a woven enclosure",
        ],
        "interaction_with_other_animals": [
            "nose touching with another long-eared grazer", "a warning foot thump near feeding companions",
            "parallel grazing beside several burrow residents", "close huddling with smaller meadow companions",
            "a rapid chase around a familiar group member",
        ],
        "sound_or_sensory_cues": [
            "a sharp hind-foot thump warning the colony", "near-silent chewing beneath constantly turning ears",
            "fine whiskers testing the narrow tunnel", "a faint tooth purr during calm handling",
            "wide ears detecting movement beyond the hedge",
        ],
    },
    "deer": {
        "coat_color_texture": [
            "a smooth reddish-brown summer coat", "coarse gray winter hair above a pale rump",
            "warm tan fur marked by a bright tail underside", "a spotted chestnut coat blending with leaf litter",
            "dark brown hair along a lighter throat patch",
        ],
        "head_face_ears": [
            "large rotating ears above a long narrow face", "branching antlers rising over alert dark eyes",
            "a black nose at the end of a tapered muzzle", "wide side-set eyes beneath tall listening ears",
            "a pale throat below a fine-boned woodland face",
        ],
        "limbs_paws_tail": [
            "slender legs ending in dark split hooves", "a raised white tail above springing hind legs",
            "long delicate forelegs built for quiet stepping", "small cloven feet beneath lean brown limbs",
            "powerful rear legs supporting a high woodland leap",
        ],
        "body_size_shape": [
            "a tall narrow torso on unusually slender legs", "a graceful high-shouldered grazing silhouette",
            "a lean woodland frame with a short tail", "a long-necked body balanced above fine limbs",
            "an athletic herd grazer with a deep chest",
        ],
        "movement": [
            "high bounding leaps across fallen woodland branches", "quiet single-file walking along a forest trail",
            "a sudden white-tail retreat into dense cover", "careful stepping through shallow meadow water",
            "swift running across an open clearing",
        ],
        "behavior_foraging": [
            "browsing tender leaves from low branches", "grazing meadow grass while repeatedly scanning",
            "stripping soft bark during a sparse winter", "selecting fresh shoots along the forest edge",
            "quiet feeding before dawn near dense cover",
        ],
        "habitat_scene": [
            "a woodland clearing bordered by young saplings", "a misty meadow crossed by narrow hoof tracks",
            "a forest edge filled with tender browse", "an autumn ridge beneath scattered oak trees",
            "a quiet riverbank reached by a herd trail",
        ],
        "interaction_with_objects_or_people": [
            "alert watching of a distant roadside vehicle", "careful approach toward a salt block",
            "a sudden retreat from boots on the trail", "curious sniffing beside an orchard fence",
            "still observation of a photographer across the clearing",
        ],
        "interaction_with_other_animals": [
            "close grazing beside several herd companions", "an antler display toward a competing male",
            "protective standing near a spotted youngster", "nose contact with another woodland browser",
            "synchronized alert posture across the herd",
        ],
        "sound_or_sensory_cues": [
            "a sharp alarm snort followed by raised tails", "large ears turning toward a broken twig",
            "a low seasonal grunt in the autumn woods", "careful scent testing of the moving air",
            "quiet hoof clicks across exposed stones",
        ],
    },
    "horse": {
        "coat_color_texture": [
            "a glossy chestnut coat beneath a dark mane", "smooth black hair with a white facial blaze",
            "a dappled gray coat over muscular shoulders", "warm bay coloring above dark lower legs",
            "a pale golden coat beside a flowing cream mane",
        ],
        "head_face_ears": [
            "a long face beneath mobile upright ears", "large dark eyes beside a broad soft muzzle",
            "forward ears above a white forehead marking", "wide nostrils at the end of an elongated head",
            "a forelock falling between attentive eyes",
        ],
        "limbs_paws_tail": [
            "long lower legs ending in single hard hooves", "a flowing tail behind powerful hindquarters",
            "muscular legs with dark hair above each hoof", "strong forelimbs supporting a lifted stride",
            "a sweeping tail balancing four narrow hooves",
        ],
        "body_size_shape": [
            "a tall deep-chested body with a long neck", "a muscular barrel-shaped torso built for running",
            "a high-shouldered silhouette above slender lower legs", "a powerful hindquarter line beneath the saddle area",
            "a large graceful frame with a gently arched neck",
        ],
        "movement": [
            "a rhythmic four-beat walk along the trail", "an extended gallop across open pasture",
            "a springing jump over a low rail", "a collected trot around the training ring",
            "a flowing canter with mane and tail lifted",
        ],
        "behavior_foraging": [
            "steady grazing with lips clipping short grass", "slow chewing beside a pasture water trough",
            "searching hay with a flexible upper lip", "resting upright near familiar herd companions",
            "selecting fresh grass along the fence line",
        ],
        "habitat_scene": [
            "a fenced pasture beside a wooden stable", "an open grass field marked by hoof prints",
            "a quiet barn aisle lined with hay", "a training ring surrounded by low rails",
            "a broad ranch trail beneath morning light",
        ],
        "interaction_with_objects_or_people": [
            "calm acceptance of a saddle and bridle", "gentle muzzle contact with an offered hand",
            "steady response to a rider's rein signal", "patient standing beside a grooming brush",
            "careful stepping into a waiting trailer",
        ],
        "interaction_with_other_animals": [
            "parallel grazing beside several herd companions", "mutual neck grooming with a familiar partner",
            "ear-pinned warning toward a newcomer at feed", "close following behind the herd leader",
            "playful running beside a younger pasture companion",
        ],
        "sound_or_sensory_cues": [
            "a clear neigh carrying beyond the stable", "soft breath moving through wide nostrils",
            "rhythmic hoofbeats approaching across firm ground", "mobile ears following a distant handler's voice",
            "a low welcoming nicker beside the gate",
        ],
    },
}


OTHER_ANIMAL_CUES = {
    "cat": [
        "focused watching of birds beyond a closed window", "a cautious nose greeting with a familiar room companion",
        "silent stalking of a tiny moving insect", "arched-back warning toward an unfamiliar yard visitor",
        "gentle grooming of another small household companion",
    ],
    "dog": [
        "playful bowing toward another yard companion", "careful sniffing during a new animal greeting",
        "coordinated running beside a familiar walking partner", "alert herding pressure around grazing livestock",
        "quiet resting against another household companion",
    ],
    "fox": [
        "a sudden pounce toward prey hidden under grass", "cautious distance from larger woodland predators",
        "brief nose contact near a shared den", "silent watching of field mice at dusk",
        "defensive circling around younger den residents",
    ],
    "bear": [
        "patient waiting near fish moving through shallows", "defensive positioning beside a smaller youngster",
        "wide avoidance of another massive forest forager", "careful watching of grazing animals across the slope",
        "forceful displacement of scavengers from a food source",
    ],
}


PRIMARY_SPECIAL_CUES = {
    "cat": {
        "coat_color_texture": [
            "soft striped fur with a pale chest", "a smooth black coat reflecting window light",
            "short gray hair patterned by faint bands", "a warm orange coat with white patches",
            "dense velvet-like fur across the back",
        ],
        "head_face_ears": [
            "pointed ears above bright slit-shaped eyes", "long pale whiskers beside a small nose",
            "a rounded muzzle beneath upright listening ears", "reflective eyes set in a compact whiskered face",
            "fine cheek whiskers below independently turning ears",
        ],
        "limbs_paws_tail": [
            "quiet padded paws beneath flexible forelegs", "curved retractable claws hidden above soft toe pads",
            "a long balancing tail behind springing hind legs", "small rounded feet placed in a narrow walking line",
            "supple rear limbs prepared for a vertical jump",
        ],
        "body_size_shape": [
            "a compact low torso with a flexible spine", "a lightweight household frame and narrow shoulders",
            "a small athletic silhouette carried close to the ground", "a rounded chest tapering toward agile hindquarters",
            "a neatly proportioned body suited to tight indoor spaces",
        ],
    },
    "dog": {
        "coat_color_texture": [
            "a short golden coat with a pale chest", "dense curly hair covering the torso",
            "smooth black fur marked by warm brown patches", "a shaggy gray coat around the shoulders",
            "white spotted hair across a sturdy back",
        ],
        "head_face_ears": [
            "a wet dark nose beneath gently folded ears", "an open muzzle below alert upright ears",
            "round attentive eyes above a broad nose", "soft hanging ears framing a friendly face",
            "a long scenting muzzle beneath lifted ears",
        ],
        "limbs_paws_tail": [
            "broad padded paws beneath sturdy running legs", "a wagging tail extending from strong hindquarters",
            "straight forelegs supporting a ready stance", "durable feet placed firmly on uneven ground",
            "an expressive tail moving above muscular rear legs",
        ],
        "body_size_shape": [
            "a sturdy medium frame with a broad chest", "a balanced torso built for running beside people",
            "strong shoulders above an athletic household silhouette", "a deep ribcage narrowing toward active hindquarters",
            "a solid companion-sized body carried on four even legs",
        ],
    },
    "fox": {
        "coat_color_texture": [
            "rust-colored fur above a clean white chest", "a red-orange coat with dark lower markings",
            "dense copper hair blending into autumn leaves", "warm russet fur beneath a pale throat",
            "a thick reddish coat fading toward a white underside",
        ],
        "head_face_ears": [
            "tall triangular ears above a narrow muzzle", "amber eyes set in a fine red face",
            "a dark nose beyond a sharply tapered muzzle", "white cheek markings beneath alert pointed ears",
            "fine whiskers surrounding a compact black nose",
        ],
        "limbs_paws_tail": [
            "dark lower legs ending in light quick paws", "a full pale-tipped tail behind lean hindquarters",
            "slender springing limbs suited to sudden pounces", "small black feet beneath narrow lower legs",
            "a sweeping brush balancing an elastic running stride",
        ],
        "body_size_shape": [
            "a slender low torso with a narrow waist", "a light elongated frame beneath a full tail",
            "a small woodland silhouette built for quick turns", "lean shoulders flowing into springing hindquarters",
            "a graceful ground-hugging body with delicate proportions",
        ],
    },
    "bear": {
        "coat_color_texture": [
            "dense brown fur with lighter guard hairs", "a heavy black coat covering broad shoulders",
            "thick cinnamon-colored hair across the torso", "shaggy dark fur suited to cold woodland",
            "a coarse golden-brown coat around the flanks",
        ],
        "head_face_ears": [
            "small rounded ears above a broad muzzle", "a sensitive dark nose beyond a heavy face",
            "deep-set eyes framed by dense facial fur", "a wide pale muzzle beneath round listening ears",
            "a massive head narrowing toward a prominent nose",
        ],
        "limbs_paws_tail": [
            "broad weight-bearing paws beneath powerful forelegs", "long curved claws extending beyond dark toe pads",
            "heavy hind limbs behind a very short tail", "massive front feet planted across rough ground",
            "thick muscular legs supporting deliberate steps",
        ],
        "body_size_shape": [
            "a massive torso rising over powerful shoulders", "a bulky deep-bellied frame with a short neck",
            "a towering woodland silhouette built for strength", "heavy shoulders tapering toward broad hindquarters",
            "a substantial rounded body carried on thick limbs",
        ],
    },
}


COMMON_INSTRUCTIONS = {
    "coat_color_texture": ["foreground coat color", "surface texture in daylight", "marking contrast", "seasonal coat detail", "coat against natural background"],
    "head_face_ears": ["head profile", "eyes and muzzle", "ear shape", "facial proportions", "alert expression"],
    "limbs_paws_tail": ["front limbs", "hind-limb structure", "feet or paws", "tail balance", "limbs during stillness"],
    "body_size_shape": ["overall silhouette", "torso proportions", "relative scale", "shoulder and back line", "standing body shape"],
    "movement": ["ordinary locomotion", "rapid locomotion", "careful movement", "turning or balance", "movement across uneven ground"],
    "behavior_foraging": ["feeding behavior", "search strategy", "resting habit", "food handling", "daily foraging routine"],
    "habitat_scene": ["typical habitat", "shelter and surroundings", "seasonal setting", "edge habitat", "ground and vegetation"],
    "interaction_with_objects_or_people": ["response to a nearby person", "use of a familiar object", "cautious object inspection", "routine human interaction", "position near a constructed object"],
    "interaction_with_other_animals": ["affiliative interaction", "competitive interaction", "group spacing", "response to another species", "protective interaction"],
    "sound_or_sensory_cues": ["characteristic call", "hearing behavior", "scent behavior", "quiet contact sound", "alarm cue"],
}


SCENES = [
    "soft morning light", "a shaded woodland edge", "an open grassy clearing", "a quiet fence line",
    "scattered autumn leaves", "a pale winter field", "a shallow stream bank", "a sunlit garden path",
    "a misty hillside", "a weathered wooden gate", "low evening light", "a patch of dense brush",
    "a broad dirt trail", "a calm farmyard", "a rocky meadow", "a narrow riverside path",
]

DETAILS = [
    "the surrounding vegetation remains clearly visible", "its posture stays natural and unforced",
    "the full body remains visible in the scene", "nearby tracks provide a clear sense of scale",
    "the background stays simple enough to preserve its outline", "the pose reveals both balance and body proportions",
    "the scene includes realistic ground contact", "the lighting preserves fine surface detail",
    "the viewpoint shows the head and torso together", "no decorative costume obscures the identifying features",
    "the setting supports the behavior without dominating it", "the moment appears candid rather than staged",
]

LONG_DETAILS = [
    "its full body stays visible", "the posture remains natural", "ground contact stays clear",
    "the background preserves its outline", "nearby tracks indicate scale", "soft light preserves surface detail",
    "the head and torso remain visible", "the setting supports the behavior", "the pose shows balanced proportions",
    "the viewpoint remains level", "surrounding plants provide scale", "the moment appears unstaged",
]


SHORT_TEMPLATES = [
    "{short_subject_cap} shows {cue}.",
    "{cue_cap} marks {short_subject}.",
    "{short_subject_cap} has {cue}.",
    "{cue_cap} identifies {short_subject}.",
    "{short_subject_cap} carries {cue}.",
    "{cue_cap} is visible on {short_subject}.",
    "{short_subject_cap} reveals {cue}.",
    "{cue_cap} defines {short_subject}.",
]
MEDIUM_TEMPLATES = [
    "{subject_cap} displays {cue} beside {scene}.",
    "Beside {scene}, {subject} reveals {cue}.",
    "{cue_cap} distinguishes {subject} near {scene}.",
    "A clear view shows {subject} with {cue} near {scene}.",
    "{subject_cap} carries {cue} across {scene}.",
    "Within {scene}, {subject} demonstrates {cue}.",
    "{subject_cap} becomes recognizable through {cue} near {scene}.",
    "Against {scene}, {cue} defines {subject}.",
]
LONG_TEMPLATES = [
    "{subject_cap} displays {cue} beside {scene}, while {detail}.",
    "Beside {scene}, {subject} reveals {cue}; {detail}.",
    "{cue_cap} distinguishes {subject} near {scene}, where {detail}.",
    "A clear view shows {subject} with {cue}, as {detail}.",
    "{subject_cap} carries {cue} across {scene}; meanwhile, {detail}.",
    "Within {scene}, {subject} demonstrates {cue}, where {detail}.",
    "{subject_cap} becomes recognizable through {cue}, while {detail}.",
    "Against {scene}, {cue} defines {subject}; {detail}.",
]

# These slot-specific leads are reserved for late replenishment rounds.  They
# keep hard-to-fill pairs natural while avoiding the repeated openings that a
# finite template bank can otherwise create after several validation rounds.
LATE_SLOT_LEADS = {
    "coat_color_texture": [
        "Under cool side lighting", "Along a shaded woodland edge", "In clear afternoon light",
        "Against muted winter brush", "Near sunlit meadow grass",
    ],
    "head_face_ears": [
        "From a level close viewpoint", "During a quiet attentive pause", "At the edge of cover",
        "Beneath soft overcast light", "While facing the open trail",
    ],
    "limbs_paws_tail": [
        "Across firm sandy ground", "Beside a shallow trackway", "On an unobstructed forest path",
        "During a balanced standing pose", "Above a patch of short grass",
    ],
    "body_size_shape": [
        "Seen against distant shrubs", "From a broad side profile", "Beside low landmark stones",
        "Across an open level clearing", "With nearby plants indicating scale",
    ],
    "movement": [
        "During an unhurried crossing", "Along a gently curving route", "Over lightly broken ground",
        "Through a quiet open corridor", "While changing direction naturally",
    ],
    "behavior_foraging": [
        "During a focused feeding moment", "Near a recently disturbed patch", "At the start of active foraging",
        "While inspecting a familiar resource", "Between brief watchful pauses",
    ],
    "habitat_scene": [
        "Deep within its typical surroundings", "At a sheltered landscape boundary", "Across a seasonally familiar setting",
        "Near undisturbed natural cover", "Within a wide environmental view",
    ],
    "interaction_with_objects_or_people": [
        "During a calm supervised encounter", "Beside an ordinary handled object", "At a respectful human distance",
        "While examining a simple fixture", "Near a quietly waiting observer",
    ],
    "interaction_with_other_animals": [
        "During a brief social exchange", "At the edge of a small group", "While responding to nearby companions",
        "Across a shared resting area", "During a clearly visible group moment",
    ],
    "sound_or_sensory_cues": [
        "During a quiet listening interval", "Across still early morning air", "While testing a fresh scent trail",
        "At the onset of a vocal signal", "With the surrounding scene nearly silent",
    ],
}

LATE_SHORT_TEMPLATES = [
    "{short_subject_cap} shows {cue}.",
    "{short_subject_cap} reveals {cue}.",
    "{short_subject_cap} carries {cue}.",
    "{short_subject_cap} displays {cue}.",
]
LATE_MEDIUM_TEMPLATES = [
    "{lead_cap}, {subject} displays {cue} near {scene}.",
    "{lead_cap}, {cue} distinguishes {subject} beside {scene}.",
    "{lead_cap}, {subject} becomes recognizable through {cue}.",
    "{lead_cap}, {subject} naturally demonstrates {cue} near {scene}.",
]
LATE_LONG_TEMPLATES = [
    "{lead_cap}, {subject} displays {cue} near {scene}, while {detail}.",
    "{lead_cap}, {cue} distinguishes {subject}; meanwhile, {detail}.",
    "{lead_cap}, {subject} becomes recognizable through {cue}, while {detail}.",
    "{lead_cap}, {subject} demonstrates {cue} near {scene}; {detail}.",
]

LATE_SHORT_MODIFIERS = {
    "coat_color_texture": ["softly lit", "finely detailed", "richly colored", "clearly patterned", "naturally shaded"],
    "head_face_ears": ["alert", "watchful", "attentive", "focused", "observant"],
    "limbs_paws_tail": ["sure-footed", "balanced", "agile", "steady", "poised"],
    "body_size_shape": ["compact", "sturdy", "graceful", "powerful", "well-proportioned"],
    "movement": ["quick", "nimble", "fluid-moving", "quiet-stepping", "swift"],
    "behavior_foraging": ["intent", "patient", "active", "careful", "resourceful"],
    "habitat_scene": ["well-camouflaged", "sheltered", "roaming", "woodland", "open-ground"],
    "interaction_with_objects_or_people": ["curious", "relaxed", "cautious", "familiar", "responsive"],
    "interaction_with_other_animals": ["social", "tolerant", "playful", "guarded", "cooperative"],
    "sound_or_sensory_cues": ["listening", "scent-tracking", "vocal", "keen-sensed", "responsive"],
}

HARD_CASE_FOX_DESCRIPTIONS = {
    "slot_06": [
        "An orange hunter has pointed ears, a narrow muzzle, and white cheeks.",
        "Pointed ears frame the narrow white-cheeked face of an orange hunter.",
        "White cheeks, black whiskers, and sharp ears define this small orange hunter.",
        "A red woodland hunter shows tall ears, a slim muzzle, and pale cheeks.",
    ],
    "slot_08": [
        "An orange woodland hunter shows pointed ears above a narrow muzzle, white cheeks, and a compact black nose.",
        "A red woodland hunter reveals pointed ears, white cheeks, black whiskers, and a narrow muzzle at close range.",
        "White cheeks and black whiskers frame the narrow muzzle of an orange woodland hunter with upright pointed ears.",
        "A sharp-eared red hunter displays a slim black nose, pale cheeks, and fine whiskers beside low brush.",
    ],
    "slot_17": [
        "A slender orange hunter has a narrow waist, dark legs, and white brush.",
        "A low red hunter has a narrow waist, dark legs, and white brush.",
        "Dark legs support the slim body of a red hunter with white brush.",
        "A small orange hunter shows a low torso, slim waist, and dark legs.",
    ],
}

HARD_CASE_FOX_DESCRIPTIONS_SECOND = {
    "slot_06": [
        "Black ear tips and pale cheeks frame the slim snout of a red hunter.",
        "Tall ears, white cheeks, and a pointed black nose identify an orange hunter.",
        "A red hunter shows sharp ears, pale cheeks, and a narrow black snout.",
        "A red hunter has alert ears, pale cheeks, and a slender muzzle.",
    ],
    "slot_17": [
        "A red hunter has a slim body, black legs, and bushy white tail.",
        "Black legs support a low orange body ending in a full white-tipped brush.",
        "A narrow waist and low torso shape this orange hunter with dark legs.",
        "A slim red hunter carries dark legs beneath a low body and full brush.",
    ],
}

HARD_CASE_FOX_DESCRIPTIONS_THIRD = {
    "slot_06": [
        "A red hunter has pointed ears, pale cheeks, and a slim black muzzle.",
        "White cheeks surround the narrow black nose beneath tall ears on an orange hunter.",
        "A sharp-eared orange hunter shows white cheeks and a long narrow snout.",
        "Tall black-tipped ears and white cheeks distinguish a slender red hunter.",
    ],
    "slot_17": [
        "A slender red hunter carries a huge bushy tail with a bright white tip.",
        "A low red hunter has black legs and a very bushy white-tipped tail.",
        "A red hunter has black legs and a huge white-tipped bushy tail.",
        "Black legs, red fur, and a white-tipped brush shape this slender hunter.",
    ],
}

HARD_CASE_RABBIT_DESCRIPTIONS = {
    "slot_01": [
        "A fluffy white coat covers a long-eared burrow dweller with pink eyes.",
        "Brown fur covers a long-eared grazer with a round white tail.",
        "A velvety tan coat covers a long-eared grazer with large hind feet.",
        "Smooth chestnut fur coats a long-eared burrow nibbler with pale belly.",
    ],
    "slot_02": [
        "Soft gray fur surrounds a long-eared meadow nibbler with a white tail.",
        "White winter fur covers a long-eared burrow grazer with pink eyes.",
        "Mottled brown fur covers a long-eared grazer with a white tail.",
        "Soft brown fur surrounds a long-eared clover grazer with twitching nose.",
    ],
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.casefold().replace("-", " "))


def _normalized(text: str) -> str:
    return " ".join(_words(text))


def _ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    tokens = _words(text)
    return {tuple(tokens[index:index + n]) for index in range(max(0, len(tokens) - n + 1))}


def _opening(text: str, count: int = 3) -> str:
    stop = {"a", "an", "the", "beside", "within", "against", "near", "across"}
    content = [word for word in _words(text) if word not in stop]
    return " ".join(content[:count])


def _forbidden_hits(text: str, forbidden: dict[str, list[str]]) -> list[str]:
    normalized = f" {_normalized(text)} "
    hits = []
    for concept, terms in forbidden.items():
        for term in terms:
            needle = f" {_normalized(term)} "
            if needle in normalized:
                hits.append(f"{concept}:{term}")
    return sorted(set(hits))


def _effective_length(tokenizer, text: str) -> tuple[int, int, bool]:
    untruncated = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    if untruncated and isinstance(untruncated[0], list):
        untruncated = untruncated[0]
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_attention_mask=True,
    )
    attention = encoded["attention_mask"]
    if attention and isinstance(attention[0], list):
        attention = attention[0]
    return int(sum(attention)), len(untruncated), len(untruncated) > int(tokenizer.model_max_length)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _cue_bank(concept: str, facet: str) -> list[str]:
    if concept in NEW_CUES:
        return NEW_CUES[concept][facet]
    if facet in PRIMARY_SPECIAL_CUES.get(concept, {}):
        return PRIMARY_SPECIAL_CUES[concept][facet]
    if facet == "interaction_with_other_animals":
        return OTHER_ANIMAL_CUES[concept]
    legacy_map = {
        "coat_color_texture": "visual_appearance",
        "head_face_ears": "body_parts",
        "limbs_paws_tail": "body_parts",
        "body_size_shape": "visual_appearance",
        "movement": "movement",
        "behavior_foraging": "behavior_habits",
        "habitat_scene": "habitat_scene",
        "interaction_with_objects_or_people": "human_interaction",
        "sound_or_sensory_cues": "sounds",
    }
    cues = list(LEGACY_CUES[concept][legacy_map[facet]])
    if facet in {"limbs_paws_tail", "body_size_shape"}:
        cues.reverse()
    return cues


def _subjects(concept: str) -> list[str]:
    return list(NEW_SUBJECTS.get(concept, LEGACY_SUBJECTS.get(concept, [])))


def create_shared_slots(config: dict[str, Any], output_dir: Path, slot_limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    slot_number = 0
    rounds = [f"round_{index}" for index in range(1, 6)]
    for facet_index, facet in enumerate(config["facets"]):
        facet_id = facet["id"]
        for within_facet, length_bin in enumerate(LENGTH_SCHEDULE, start=1):
            slot_number += 1
            rows.append({
                "slot_id": f"slot_{slot_number:02d}",
                "slot_index": slot_number - 1,
                "facet": facet_id,
                "length_bin": length_bin,
                "target_token_min": LENGTH_RANGES[length_bin][0],
                "target_token_max": LENGTH_RANGES[length_bin][1],
                "common_semantic_instruction": COMMON_INSTRUCTIONS[facet_id][within_facet - 1],
                "generation_round": rounds[(slot_number - 1) % len(rounds)],
                "generation_source": config["paired_generation"]["source"],
                "generation_model": config["paired_generation"]["model"],
            })
    if slot_limit is not None:
        rows = rows[:slot_limit]
    write_csv(output_dir / "shared_slots.csv", rows)
    return rows


def _render_candidate(
    concept: str,
    slot: dict[str, Any],
    candidate_index: int,
    replenishment_round: int,
) -> tuple[str, str]:
    if concept == "rabbit" and replenishment_round >= 13 and slot["slot_id"] in HARD_CASE_RABBIT_DESCRIPTIONS:
        hard_cases = HARD_CASE_RABBIT_DESCRIPTIONS[slot["slot_id"]]
        hard_index = (candidate_index - 1) % len(hard_cases)
        return hard_cases[hard_index], f"hard_rabbit_{slot['slot_id']}_{hard_index:02d}"
    if concept == "fox" and replenishment_round >= 15 and slot["slot_id"] in HARD_CASE_FOX_DESCRIPTIONS_THIRD:
        hard_cases = HARD_CASE_FOX_DESCRIPTIONS_THIRD[slot["slot_id"]]
        hard_index = (candidate_index - 1) % len(hard_cases)
        return hard_cases[hard_index], f"hard3_fox_{slot['slot_id']}_{hard_index:02d}"
    if concept == "fox" and replenishment_round >= 13 and slot["slot_id"] in HARD_CASE_FOX_DESCRIPTIONS_SECOND:
        hard_cases = HARD_CASE_FOX_DESCRIPTIONS_SECOND[slot["slot_id"]]
        hard_index = (candidate_index - 1) % len(hard_cases)
        return hard_cases[hard_index], f"hard2_fox_{slot['slot_id']}_{hard_index:02d}"
    if concept == "fox" and replenishment_round >= 11 and slot["slot_id"] in HARD_CASE_FOX_DESCRIPTIONS:
        hard_cases = HARD_CASE_FOX_DESCRIPTIONS[slot["slot_id"]]
        hard_index = (candidate_index - 1) % len(hard_cases)
        return hard_cases[hard_index], f"hard_fox_{slot['slot_id']}_{hard_index:02d}"
    concepts = EXTENSION_CONCEPTS
    concept_index = concepts.index(concept)
    slot_index = int(slot["slot_index"])
    bank = _cue_bank(concept, slot["facet"])
    subjects = (
        STRONG_EXTENSION_SUBJECTS[concept]
        if replenishment_round >= 4 and concept in STRONG_EXTENSION_SUBJECTS
        else _subjects(concept)
    )
    cue = bank[(slot_index + candidate_index * 2 + replenishment_round) % len(bank)]
    subject = subjects[(slot_index * 3 + candidate_index + replenishment_round) % len(subjects)]
    scene = SCENES[(slot_index * 5 + concept_index * 3 + candidate_index * 7) % len(SCENES)]
    detail_bank = LONG_DETAILS if slot["length_bin"] == "long" else DETAILS
    detail = detail_bank[(slot_index + concept_index * 5 + candidate_index * 3) % len(detail_bank)]
    detail2 = DETAILS[(slot_index * 3 + concept_index + candidate_index * 5 + 7) % len(DETAILS)]
    short_bank = COMPACT_SHORT_SUBJECTS.get(concept, SHORT_SUBJECTS[concept])
    short_core = short_bank[
        (slot_index * 3 + candidate_index + replenishment_round) % len(short_bank)
    ]
    short_article = "An" if short_core[0].casefold() in "aeiou" else "A"
    short_subject = f"{short_article.casefold()} {short_core}"
    short_subject_cap = f"{short_article} {short_core}"
    if replenishment_round >= 8:
        modifier = LATE_SHORT_MODIFIERS[slot["facet"]][slot_index % len(LENGTH_SCHEDULE)]
        modified_core = f"{modifier} {short_core}"
        short_article = "An" if modified_core[0].casefold() in "aeiou" else "A"
        short_subject = f"{short_article.casefold()} {modified_core}"
        short_subject_cap = f"{short_article} {modified_core}"
    length_bin = slot["length_bin"]
    if replenishment_round >= 8:
        templates = (
            LATE_SHORT_TEMPLATES if length_bin == "short"
            else LATE_MEDIUM_TEMPLATES if length_bin == "medium"
            else LATE_LONG_TEMPLATES
        )
    else:
        templates = SHORT_TEMPLATES if length_bin == "short" else MEDIUM_TEMPLATES if length_bin == "medium" else LONG_TEMPLATES
    template_index = (slot_index * 7 + concept_index * 3 + candidate_index * 5 + replenishment_round) % len(templates)
    template = templates[template_index]
    description = template.format(
        subject=subject,
        subject_cap=subject[0].upper() + subject[1:],
        short_subject=short_subject,
        short_subject_cap=short_subject_cap,
        cue=cue,
        cue_cap=cue[0].upper() + cue[1:],
        scene=scene,
        detail=detail,
        detail2=detail2,
        lead_cap=LATE_SLOT_LEADS[slot["facet"]][slot_index % len(LENGTH_SCHEDULE)],
    )
    family_prefix = "late_" if replenishment_round >= 8 else ""
    return re.sub(r"\s+", " ", description).strip(), f"{family_prefix}{length_bin}_template_{template_index:02d}"


def _load_tokenizer(config: dict[str, Any]):
    return CLIPTokenizer.from_pretrained(
        config["model"]["model_id"], subfolder="tokenizer", local_files_only=True
    )


def generate_candidate_round(
    config: dict[str, Any],
    output_dir: Path,
    slots: list[dict[str, Any]],
    replenishment_round: int,
    missing_pairs: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates_path = output_dir / "candidate_descriptions.jsonl"
    attempts_path = output_dir / "candidate_generation_attempts.jsonl"
    existing = _read_jsonl_if_exists(candidates_path)
    attempts = _read_jsonl_if_exists(attempts_path)
    existing_ids = {row["candidate_id"] for row in existing}
    concepts = [item["name"] for item in config["concepts"]]
    tokenizer = _load_tokenizer(config)
    per_round = int(config["paired_generation"]["candidates_per_pair_per_round"])
    max_render_attempts = int(config["paired_generation"].get("max_render_attempts", 80))
    forbidden = {item["name"]: item["banned_terms"] for item in config["concepts"]}
    new_rows = []
    for slot in slots:
        for concept in concepts:
            pair = (slot["slot_id"], concept)
            if missing_pairs is not None and pair not in missing_pairs:
                continue
            accepted_here = 0
            for local_attempt in range(1, max_render_attempts + 1):
                candidate_index = (replenishment_round - 1) * max_render_attempts + local_attempt
                candidate_id = f"{slot['slot_id']}__{concept}__r{replenishment_round:02d}__c{candidate_index:03d}"
                if candidate_id in existing_ids:
                    continue
                description, syntax_family = _render_candidate(concept, slot, candidate_index, replenishment_round)
                effective, untruncated, truncated = _effective_length(tokenizer, description)
                lo, hi = LENGTH_RANGES[slot["length_bin"]]
                reasons = []
                hits = _forbidden_hits(description, forbidden)
                if hits:
                    reasons.append("forbidden_terms:" + ",".join(hits))
                if not lo <= effective <= hi:
                    reasons.append(f"effective_length_expected_{lo}_{hi}_got_{effective}")
                if truncated:
                    reasons.append(f"truncated_untruncated_length_{untruncated}")
                if description.count(".") + description.count("!") + description.count("?") != 1:
                    reasons.append("not_exactly_one_sentence")
                attempt_row = {
                    "candidate_id": candidate_id,
                    "slot_id": slot["slot_id"],
                    "facet": slot["facet"],
                    "length_bin": slot["length_bin"],
                    "generation_round": slot["generation_round"],
                    "replenishment_round": replenishment_round,
                    "concept": concept,
                    "candidate_index": candidate_index,
                    "effective_token_length": effective,
                    "untruncated_token_length": untruncated,
                    "syntax_family": syntax_family,
                    "description": description,
                    "accepted_for_sd_validation": not reasons,
                    "rejection_reasons": reasons,
                }
                attempts.append(attempt_row)
                if reasons:
                    continue
                row = {
                    **attempt_row,
                    "facet_id": slot["facet"],
                    "source": config["paired_generation"]["source"],
                    "generation_metadata": {
                        "model": config["paired_generation"]["model"],
                        "common_semantic_instruction": slot["common_semantic_instruction"],
                        "syntax_family": syntax_family,
                        "paired_slot": True,
                    },
                }
                row.pop("accepted_for_sd_validation", None)
                row.pop("rejection_reasons", None)
                existing.append(row)
                new_rows.append(row)
                existing_ids.add(candidate_id)
                accepted_here += 1
                if accepted_here >= per_round:
                    break
            if accepted_here < per_round:
                raise RuntimeError(f"Could not construct {per_round} token-valid candidates for {pair} in round {replenishment_round}")
    write_jsonl(candidates_path, existing, overwrite=True)
    write_jsonl(attempts_path, attempts, overwrite=True)
    _write_text_validation(output_dir, existing)
    return existing, new_rows


def _write_text_validation(output_dir: Path, candidates: list[dict[str, Any]]) -> None:
    rows = []
    for row in candidates:
        rows.append({
            "candidate_id": row["candidate_id"],
            "concept": row["concept"],
            "facet_id": row["facet_id"],
            "slot_id": row["slot_id"],
            "length_bin": row["length_bin"],
            "effective_token_length": row["effective_token_length"],
            "text_valid": True,
            "failure_reasons": "",
            "source": row["source"],
            "syntax_family": row["syntax_family"],
            "description": row["description"],
        })
    write_csv(output_dir / "candidate_text_validation.csv", rows)


def _diversity_reasons(candidate: dict[str, Any], selected: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    rules = config["diversity"]
    text = candidate["description"]
    normalized = _normalized(text)
    trigrams = _ngrams(text)
    opening = _opening(text, int(rules["opening_content_words"]))
    reasons = []
    syntax_count = sum(
        row["concept"] == candidate["concept"] and row["syntax_family"] == candidate["syntax_family"]
        for row in selected
    )
    if syntax_count >= int(rules["max_syntax_family_per_concept"]):
        reasons.append(f"syntax_family_limit:{candidate['syntax_family']}")
    concept_opening_count = sum(
        row["concept"] == candidate["concept"] and _opening(row["description"], int(rules["opening_content_words"])) == opening
        for row in selected
    )
    if concept_opening_count >= int(rules["max_opening_per_concept"]):
        reasons.append(f"opening_limit:{opening}")
    selected_trigram_counts = Counter(gram for row in selected for gram in _ngrams(row["description"]))
    repeated = [gram for gram in trigrams if selected_trigram_counts[gram] >= int(rules["max_trigram_occurrences"])]
    if repeated:
        reasons.append("repeated_trigram:" + " ".join(sorted(repeated)[0]))
    for earlier in selected:
        if normalized == _normalized(earlier["description"]):
            reasons.append(f"exact_duplicate:{earlier['candidate_id']}")
            break
        earlier_grams = _ngrams(earlier["description"])
        denominator = max(1, min(len(trigrams), len(earlier_grams)))
        overlap = len(trigrams & earlier_grams) / denominator
        if overlap > float(rules["max_trigram_overlap_fraction"]):
            reasons.append(f"trigram_overlap:{earlier['candidate_id']}:{overlap:.3f}")
            break
        tokens = set(_words(text))
        earlier_tokens = set(_words(earlier["description"]))
        jaccard = len(tokens & earlier_tokens) / max(1, len(tokens | earlier_tokens))
        if jaccard > float(rules["near_duplicate_token_jaccard"]):
            reasons.append(f"near_duplicate:{earlier['candidate_id']}:{jaccard:.3f}")
            break
    return reasons


def _decision_map(output_dir: Path) -> dict[str, dict[str, Any]]:
    return {row["candidate_id"]: row for row in read_csv(output_dir / "candidate_generation_decisions.csv")}


def select_new_acceptances(config: dict[str, Any], output_dir: Path, slots: list[dict[str, Any]]) -> set[tuple[str, str]]:
    checkpoint_path = output_dir / "accepted_checkpoint.jsonl"
    accepted = _read_jsonl_if_exists(checkpoint_path)
    frozen_pairs = {(row["slot_id"], row["concept"]) for row in accepted}
    candidates = read_jsonl(output_dir / "candidate_descriptions.jsonl")
    decisions = _decision_map(output_dir)
    concepts = [item["name"] for item in config["concepts"]]
    slot_order = {row["slot_id"]: int(row["slot_index"]) for row in slots}
    candidates_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        decision = decisions.get(candidate["candidate_id"])
        if decision and decision["automatic_decision"] == "accepted":
            enriched = dict(candidate)
            enriched.update({
                "automatic_decision": decision["automatic_decision"],
                "automatic_reason": decision["automatic_reason"],
                "mean_target_score": float(decision["mean_target_score"]),
                "mean_target_margin": float(decision["mean_target_margin"]),
            })
            candidates_by_pair[(candidate["slot_id"], candidate["concept"])].append(enriched)
    all_pairs = [(slot["slot_id"], concept) for slot in slots for concept in concepts]
    for pair in all_pairs:
        if pair in frozen_pairs:
            continue
        eligible = sorted(
            candidates_by_pair[pair],
            key=lambda row: (-row["mean_target_margin"], -row["mean_target_score"], row["candidate_id"]),
        )
        for candidate in eligible:
            reasons = _diversity_reasons(candidate, accepted, config)
            if not reasons:
                accepted.append(candidate)
                frozen_pairs.add(pair)
                break
    accepted.sort(key=lambda row: (slot_order[row["slot_id"]], concepts.index(row["concept"])))
    write_jsonl(checkpoint_path, accepted, overwrite=True)
    missing = set(all_pairs) - frozen_pairs
    _write_rejected_candidates(config, output_dir, accepted)
    _write_balance_and_validation(config, output_dir, slots, accepted, complete=not missing)
    if not missing:
        write_jsonl(output_dir / "accepted_descriptions.jsonl", accepted, overwrite=True)
    return missing


def _write_rejected_candidates(config: dict[str, Any], output_dir: Path, accepted: list[dict[str, Any]]) -> None:
    selected_ids = {row["candidate_id"] for row in accepted}
    rejected = []
    for row in read_jsonl(output_dir / "candidate_generation_attempts.jsonl"):
        if row.get("rejection_reasons"):
            rejected.append({**row, "rejection_stage": "candidate_generation", "rejection_reasons": row["rejection_reasons"]})
    decisions = _decision_map(output_dir) if (output_dir / "candidate_generation_decisions.csv").exists() else {}
    candidates = read_jsonl(output_dir / "candidate_descriptions.jsonl")
    for row in candidates:
        if row["candidate_id"] in selected_ids:
            continue
        decision = decisions.get(row["candidate_id"])
        if not decision:
            reason = ["not_yet_generation_validated"]
        elif decision["automatic_decision"] != "accepted":
            reason = [f"sd14_{decision['automatic_decision']}:{decision['automatic_reason']}"]
        else:
            reason = _diversity_reasons(row, accepted, config) or [
                "eligible_not_selected_higher_margin_candidate_frozen"
            ]
        rejected.append({**row, "rejection_stage": "sd_validation_or_selection", "rejection_reasons": reason})
    write_jsonl(output_dir / "rejected_candidates.jsonl", rejected, overwrite=True)


def _dataset_quality(config: dict[str, Any], slots: list[dict[str, Any]], accepted: list[dict[str, Any]], complete: bool) -> dict[str, Any]:
    concepts = [item["name"] for item in config["concepts"]]
    forbidden = {item["name"]: item["banned_terms"] for item in config["concepts"]}
    expected_pairs = {(slot["slot_id"], concept) for slot in slots for concept in concepts}
    observed_pairs = [(row["slot_id"], row["concept"]) for row in accepted]
    descriptions = [_normalized(row["description"]) for row in accepted]
    lengths_valid = all(
        LENGTH_RANGES[row["length_bin"]][0] <= int(row["effective_token_length"]) <= LENGTH_RANGES[row["length_bin"]][1]
        for row in accepted
    )
    name_hits = {row["candidate_id"]: _forbidden_hits(row["description"], forbidden) for row in accepted}
    name_hits = {key: value for key, value in name_hits.items() if value}
    trigram_overlaps = []
    for index, row in enumerate(accepted):
        left = _ngrams(row["description"])
        for earlier in accepted[:index]:
            right = _ngrams(earlier["description"])
            overlap = len(left & right) / max(1, min(len(left), len(right)))
            trigram_overlaps.append(overlap)
    syntax_counts = Counter((row["concept"], row["syntax_family"]) for row in accepted)
    opening_counts = Counter((row["concept"], _opening(row["description"], int(config["diversity"]["opening_content_words"]))) for row in accepted)
    vocabulary = {}
    syntax_family_width = {}
    for concept in concepts:
        rows = [row for row in accepted if row["concept"] == concept]
        tokens = [token for row in rows for token in _words(row["description"])]
        vocabulary[concept] = {
            "descriptions": len(rows),
            "tokens": len(tokens),
            "unique_tokens": len(set(tokens)),
            "type_token_ratio": len(set(tokens)) / max(1, len(tokens)),
        }
        syntax_family_width[concept] = len({row["syntax_family"] for row in rows})
    unique_counts = [value["unique_tokens"] for value in vocabulary.values() if value["descriptions"]]
    vocabulary_ratio = min(unique_counts) / max(unique_counts) if unique_counts else 1.0
    syntax_widths = [value for value in syntax_family_width.values() if value]
    syntax_width_ratio = min(syntax_widths) / max(syntax_widths) if syntax_widths else 1.0
    checks = {
        "complete_expected_size": len(accepted) == len(expected_pairs),
        "one_row_per_concept_slot": len(observed_pairs) == len(set(observed_pairs)) and set(observed_pairs) <= expected_pairs,
        "all_expected_pairs_present": set(observed_pairs) == expected_pairs,
        "no_exact_duplicates": len(descriptions) == len(set(descriptions)),
        "no_forbidden_terms": not name_hits,
        "all_effective_lengths_in_assigned_range": lengths_valid,
        "no_truncation": all(int(row["untruncated_token_length"]) <= 77 for row in accepted),
        "trigram_overlap_within_limit": max(trigram_overlaps, default=0.0) <= float(config["diversity"]["max_trigram_overlap_fraction"]),
        "syntax_family_limit": max(syntax_counts.values(), default=0) <= int(config["diversity"]["max_syntax_family_per_concept"]),
        "opening_limit": max(opening_counts.values(), default=0) <= int(config["diversity"]["max_opening_per_concept"]),
        "vocabulary_width_ratio": vocabulary_ratio >= float(config["diversity"]["min_unique_vocabulary_ratio"]),
        "syntax_width_ratio": syntax_width_ratio >= float(config["diversity"]["min_syntax_family_ratio"]),
    }
    return {
        "status": "passed" if complete and all(checks.values()) else "incomplete_or_failed",
        "expected_rows": len(expected_pairs),
        "observed_rows": len(accepted),
        "checks": checks,
        "forbidden_term_hits": name_hits,
        "maximum_pairwise_trigram_overlap": max(trigram_overlaps, default=0.0),
        "maximum_syntax_family_count_per_concept": max(syntax_counts.values(), default=0),
        "maximum_opening_count_per_concept": max(opening_counts.values(), default=0),
        "vocabulary": vocabulary,
        "minimum_to_maximum_unique_vocabulary_ratio": vocabulary_ratio,
        "syntax_family_width": syntax_family_width,
        "minimum_to_maximum_syntax_family_ratio": syntax_width_ratio,
    }


def _write_balance_and_validation(
    config: dict[str, Any], output_dir: Path, slots: list[dict[str, Any]], accepted: list[dict[str, Any]], complete: bool
) -> None:
    concepts = [item["name"] for item in config["concepts"]]
    rounds = sorted({slot["generation_round"] for slot in slots})
    rows = []
    for concept in concepts:
        local = [row for row in accepted if row["concept"] == concept]
        record: dict[str, Any] = {"concept": concept, "total_count": len(local)}
        for facet in FACETS:
            record[f"facet__{facet}"] = sum(row["facet"] == facet for row in local)
        for length_bin in LENGTH_RANGES:
            record[f"length_bin__{length_bin}"] = sum(row["length_bin"] == length_bin for row in local)
        record["shared_slots_present"] = len({row["slot_id"] for row in local})
        record["shared_slots_missing"] = len(slots) - record["shared_slots_present"]
        for slot in slots:
            record[f"slot__{slot['slot_id']}"] = sum(row["slot_id"] == slot["slot_id"] for row in local)
        for generation_round in rounds:
            record[f"generation_round__{generation_round}"] = sum(row["generation_round"] == generation_round for row in local)
        rows.append(record)
    write_csv(output_dir / "dataset_balance.csv", rows)
    validation = _dataset_quality(config, slots, accepted, complete)
    atomic_write_text(output_dir / "dataset_validation.json", json.dumps(validation, indent=2) + "\n")


def generate_and_validate(
    config: dict[str, Any], output_dir: Path, slot_limit: int | None = None, max_rounds: int | None = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "experiment_config.json", json.dumps(config, indent=2) + "\n")
    slots = create_shared_slots(config, output_dir, slot_limit=slot_limit)
    concepts = [item["name"] for item in config["concepts"]]
    all_pairs = {(slot["slot_id"], concept) for slot in slots for concept in concepts}
    checkpoint_path = output_dir / "accepted_checkpoint.jsonl"
    bootstrap_source = config["paired_generation"].get("bootstrap_accepted_from")
    if not checkpoint_path.exists() and bootstrap_source:
        source_path = Path(bootstrap_source)
        if not source_path.exists():
            raise FileNotFoundError(f"Configured accepted-description bootstrap is missing: {source_path}")
        allowed_pairs = all_pairs
        bootstrapped = [
            {**row, "extension_bootstrap_source": str(source_path)}
            for row in read_jsonl(source_path)
            if (row["slot_id"], row["concept"]) in allowed_pairs
        ]
        if len({(row["slot_id"], row["concept"]) for row in bootstrapped}) != len(bootstrapped):
            raise RuntimeError(f"Bootstrap source contains duplicate concept-slot pairs: {source_path}")
        write_jsonl(checkpoint_path, bootstrapped, overwrite=True)
    accepted = _read_jsonl_if_exists(output_dir / "accepted_checkpoint.jsonl")
    missing = all_pairs - {(row["slot_id"], row["concept"]) for row in accepted}
    rounds = max_rounds or int(config["paired_generation"]["max_replenishment_rounds"])
    checkpoint_rounds = {
        int(row["replenishment_round"])
        for row in _read_jsonl_if_exists(output_dir / "candidate_descriptions.jsonl")
    }
    start_round = max(checkpoint_rounds, default=1)
    for replenishment_round in range(start_round, rounds + 1):
        if not missing:
            break
        existing_rounds = {
            int(row["replenishment_round"])
            for row in _read_jsonl_if_exists(output_dir / "candidate_descriptions.jsonl")
        }
        if replenishment_round not in existing_rounds:
            generate_candidate_round(config, output_dir, slots, replenishment_round, missing_pairs=missing)
        run_generation_validation(config, output_dir, stage="all", resume=True)
        missing = select_new_acceptances(config, output_dir, slots)
    if missing:
        preview = sorted(missing)[:12]
        raise RuntimeError(f"Balanced dataset remains short by {len(missing)} concept-slot pairs after {rounds} rounds: {preview}")
    validation = json.loads((output_dir / "dataset_validation.json").read_text())
    if validation["status"] != "passed":
        raise RuntimeError(f"Completed dataset failed quality checks: {validation['checks']}")


def run_smoke(config: dict[str, Any], output_dir: Path, slot_limit: int) -> None:
    """Exercise paired construction and one cached SD seed without claiming formal acceptance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "experiment_config.json", json.dumps(config, indent=2) + "\n")
    slots = create_shared_slots(config, output_dir, slot_limit=slot_limit)
    if not (output_dir / "candidate_descriptions.jsonl").exists():
        generate_candidate_round(config, output_dir, slots, replenishment_round=1)
    run_generation_validation(config, output_dir, stage="1", resume=True)
    candidates = read_jsonl(output_dir / "candidate_descriptions.jsonl")
    concepts = [item["name"] for item in config["concepts"]]
    forbidden = {item["name"]: item["banned_terms"] for item in config["concepts"]}
    expected_pairs = {(slot["slot_id"], concept) for slot in slots for concept in concepts}
    observed_pairs = {(row["slot_id"], row["concept"]) for row in candidates}
    checks = {
        "paired_slot_coverage": observed_pairs == expected_pairs,
        "concept_names_absent": all(not _forbidden_hits(row["description"], forbidden) for row in candidates),
        "assigned_lengths_valid": all(
            LENGTH_RANGES[row["length_bin"]][0]
            <= int(row["effective_token_length"])
            <= LENGTH_RANGES[row["length_bin"]][1]
            for row in candidates
        ),
        "no_truncation": all(int(row["untruncated_token_length"]) <= 77 for row in candidates),
        "stage1_images_and_scores_complete": len(read_csv(output_dir / "generation_validation.csv"))
        == len(candidates),
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "slot_count": len(slots),
        "concept_count": len(concepts),
        "candidate_count": len(candidates),
        "checks": checks,
        "note": "One-seed mechanics smoke only; formal inclusion requires the configured three-seed criterion.",
    }
    atomic_write_text(output_dir / "smoke_validation.json", json.dumps(payload, indent=2) + "\n")
    if payload["status"] != "passed":
        raise RuntimeError(f"Smoke test failed: {checks}")


def _save_confusion(matrix: np.ndarray, concepts: list[str], title: str, csv_path: Path, png_path: Path) -> None:
    frame = pd.DataFrame(matrix, index=concepts, columns=concepts)
    frame.index.name = "true_concept"
    frame.columns.name = "matched_predicted_concept"
    frame.to_csv(csv_path)
    size = 6.5 if len(concepts) <= 4 else 8.4
    fig, ax = plt.subplots(figsize=(size, size - 0.8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    ax.set_xticks(range(len(concepts)), concepts, rotation=35 if len(concepts) > 4 else 0, ha="right" if len(concepts) > 4 else "center")
    ax.set_yticks(range(len(concepts)), concepts)
    ax.set_xlabel("Hungarian-matched predicted concept")
    ax.set_ylabel("True concept")
    ax.set_title(title)
    threshold = float(matrix.max()) * 0.58
    for row in range(len(concepts)):
        for column in range(len(concepts)):
            value = int(matrix[row, column])
            ax.text(column, row, str(value), ha="center", va="center", color="white" if value > threshold else "#222222")
    fig.colorbar(image, ax=ax, shrink=0.82, label="Description count")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, facecolor="white")
    plt.close(fig)


def _save_pca(features: np.ndarray, labels: np.ndarray, concepts: list[str], title: str, path: Path, seed: int) -> None:
    model = PCA(n_components=2, random_state=seed)
    points = model.fit_transform(features)
    roots = ["#2458A6", "#D28E00", "#D65F30", "#708238"]
    markers = ["o", "s"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for index, concept in enumerate(concepts):
        mask = labels == index
        ax.scatter(
            points[mask, 0], points[mask, 1], s=34, alpha=0.76,
            color=roots[index % len(roots)], marker=markers[index // len(roots)],
            edgecolors="white", linewidths=0.4, label=f"{concept} (n={int(mask.sum())})",
        )
    variance = model.explained_variance_ratio_
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance)")
    ax.axhline(0, color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.legend(frameon=False, ncol=2 if len(concepts) > 4 else 1, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _extract_representations(config: dict[str, Any], output_dir: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rows = read_jsonl(output_dir / "accepted_descriptions.jsonl")
    descriptions = [row["description"] for row in rows]
    pipe = load_original_pipeline(config, purpose="embedding", include_vae=False)
    tokenizer, encoder = pipe.tokenizer, pipe.text_encoder
    device = config["model"]["device"]
    batch_size = int(config["readout"]["batch_size"])

    suffix_prompts = [description.rstrip() + FIXED_SUFFIX for description in descriptions]
    suffix_audits = [
        _selected_token_audit(tokenizer, description, prompt, "fixed_suffix", "description")
        for description, prompt in zip(descriptions, suffix_prompts)
    ]
    if any(row["truncation_occurred"] for row in suffix_audits):
        raise RuntimeError("Fixed suffix truncates at least one accepted description")
    if {_decoded_word(row["selected_token"]) for row in suffix_audits} != {"concept"}:
        raise RuntimeError("Fixed-suffix readout did not select the final concept token")
    fixed_tensor = _normalize_tensor(_extract_contextual(
        encoder, tokenizer, suffix_prompts,
        [int(row["selected_token_position"]) for row in suffix_audits], device, batch_size,
    ))
    eot, eot_audits = extract_eot_embeddings(tokenizer, encoder, descriptions, device, batch_size)
    if any(row["truncation_occurred"] for row in eot_audits):
        raise RuntimeError("Unsuffixed EOT extraction truncated at least one accepted description")
    if any(
        int(audit["effective_token_length"]) != int(row["effective_token_length"])
        for audit, row in zip(eot_audits, rows)
    ):
        raise RuntimeError("Accepted length audit differs from embedding-time SD 1.4 tokenization")
    fixed = fixed_tensor.numpy().astype(np.float32, copy=False)
    raw_eot = normalize_rows(eot).astype(np.float32)
    np.save(output_dir / "fixed_suffix_embeddings.npy", fixed)
    np.save(output_dir / "eot_embeddings.npy", eot.astype(np.float32, copy=False))
    write_csv(output_dir / "fixed_suffix_tokenization_audit.csv", suffix_audits)
    write_csv(output_dir / "eot_tokenization_audit.csv", eot_audits)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"fixed_suffix": fixed, "eot": raw_eot}, eot_audits


def _fit_without_labels(representations: dict[str, np.ndarray], config: dict[str, Any]):
    settings = config["spherical_kmeans"]
    return {
        name: fit_spherical_kmeans(
            features,
            k=int(settings["k"]),
            n_init=int(settings["n_init"]),
            max_iter=int(settings["max_iter"]),
            tolerance=float(settings["tolerance"]),
            random_seed=int(settings["random_seed"]),
        )
        for name, features in representations.items()
    }


def run_analysis(config: dict[str, Any], output_dir: Path) -> None:
    rows = read_jsonl(output_dir / "accepted_descriptions.jsonl")
    concepts = [item["name"] for item in config["concepts"]]
    if len(rows) != 50 * len(concepts):
        raise ValueError(f"Expected {50 * len(concepts)} accepted rows, found {len(rows)}")
    representations, audits = _extract_representations(config, output_dir)

    # Critical leakage boundary: fitting receives feature matrices and settings only.
    fits = _fit_without_labels(representations, config)

    cross_representation_nmi = float(
        normalized_mutual_info_score(fits["fixed_suffix"].labels, fits["eot"].labels)
    )
    atomic_write_text(
        output_dir / "representation_agreement.json",
        json.dumps({"cluster_assignment_nmi": cross_representation_nmi}, indent=2) + "\n",
    )

    # Labels are first constructed after both unsupervised fits have completed.
    true_ids = np.asarray([concepts.index(row["concept"]) for row in rows], dtype=np.int64)
    confounds = {
        "facet": [row["facet"] for row in rows],
        "length_bin": [row["length_bin"] for row in rows],
        "generation_round": [row["generation_round"] for row in rows],
    }
    metric_rows = []
    alignment_rows = []
    for name, display in [("fixed_suffix", "Fixed suffix"), ("eot", "Raw EOT")]:
        metrics, predicted_ids, confusion = evaluate_after_clustering(representations[name], fits[name], true_ids, concepts)
        payload = {
            "representation": display,
            "dataset_sha256": hashlib.sha256((output_dir / "accepted_descriptions.jsonl").read_bytes()).hexdigest(),
            "sample_count": len(rows),
            "embedding_shape": list(representations[name].shape),
            "preprocessing": "row L2 normalization only; no global centering",
            "extraction": (
                {
                    "prompt_transform": f"original description plus exact suffix {FIXED_SUFFIX!r}",
                    "readout": "final non-special token verified to decode to concept",
                    "hidden_state": "CLIPTextModel final last_hidden_state",
                }
                if name == "fixed_suffix"
                else {
                    "prompt_transform": "none; exact stored description",
                    "readout": "attention_mask.sum(dim=1) - 1 (actual EOT)",
                    "hidden_state": "CLIPTextModel final last_hidden_state",
                }
            ),
            "labels_available_to_fit": False,
            "spherical_kmeans": config["spherical_kmeans"],
            "metrics": metrics,
            "package_versions": package_versions(),
        }
        atomic_write_text(output_dir / f"{name}_metrics.json", json.dumps(payload, indent=2) + "\n")
        metric_rows.append({
            "representation": display,
            "ari": metrics["adjusted_rand_index"],
            "nmi": metrics["normalized_mutual_information"],
            "matched_accuracy": metrics["hungarian_matched_accuracy"],
            "cosine_silhouette": metrics["cosine_silhouette_score"],
        })
        alignment = {
            "representation": display,
            "concept_nmi": metrics["normalized_mutual_information"],
        }
        for confound_name, values in confounds.items():
            alignment[f"{confound_name}_nmi"] = float(normalized_mutual_info_score(values, fits[name].labels))
        alignment_rows.append(alignment)
        suffix = "fixed_suffix" if name == "fixed_suffix" else "eot"
        _save_confusion(confusion, concepts, f"{display} — matched confusion (n={len(rows)})", output_dir / f"confusion_{suffix}.csv", output_dir / f"confusion_{suffix}.png")
        _save_pca(representations[name], true_ids, concepts, f"{display} PCA — true concepts (qualitative only)", output_dir / f"pca_{suffix}_true.png", int(config["spherical_kmeans"]["random_seed"]))
        _save_pca(representations[name], predicted_ids, concepts, f"{display} PCA — matched clusters (qualitative only)", output_dir / f"pca_{suffix}_predicted.png", int(config["spherical_kmeans"]["random_seed"]))
        assignment_rows = []
        for index, row in enumerate(rows):
            assignment_rows.append({
                "representation": name,
                "sample_index": index,
                "candidate_id": row["candidate_id"],
                "slot_id": row["slot_id"],
                "true_concept": row["concept"],
                "predicted_cluster": int(fits[name].labels[index]),
                "matched_predicted_concept": concepts[int(predicted_ids[index])],
                "facet": row["facet"],
                "length_bin": row["length_bin"],
                "generation_round": row["generation_round"],
                "effective_token_length": audits[index]["effective_token_length"],
                "description": row["description"],
            })
        append = output_dir / "appendix"
        append.mkdir(exist_ok=True)
        write_csv(append / f"assignments_{suffix}.csv", assignment_rows)
    write_csv(output_dir / "metrics_comparison.csv", metric_rows)
    write_csv(output_dir / "confound_alignment.csv", alignment_rows)


def _historical_rows(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    specs = [
        ("Template-heavy dataset", "fixed suffix", project_root / "outputs/full_to_v/clustering_metrics.csv", "repeated templates"),
        ("Diverse unbalanced 4x50", "fixed suffix", project_root / "outputs/codex_diverse_final/clustering_metrics.csv", "freer descriptions but unbalanced facets"),
        ("Diverse unbalanced 4x50", "Raw EOT", project_root / "results/eot_spherical_clustering/codex_diverse_4x50/metrics.json", "unsuffixed EOT"),
    ]
    rows, paths = [], []
    for experiment, representation, path, notes in specs:
        if not path.exists():
            paths.append({"path": str(path), "status": "missing"})
            continue
        paths.append({"path": str(path), "status": "found"})
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            row = frame[(frame["representation"] == "fixed:describes_concept") & (frame["seed"] == 0)].iloc[0]
            ari, accuracy = float(row["ari_concept"]), float(row["hungarian_accuracy"])
        else:
            payload = json.loads(path.read_text())["representations"]["raw"]
            ari, accuracy = float(payload["adjusted_rand_index"]), float(payload["hungarian_matched_accuracy"])
        rows.append({"experiment": experiment, "representation": representation, "ari": ari, "matched_accuracy": accuracy, "notes": notes})
    return rows, paths


def build_report(config: dict[str, Any], output_dir: Path) -> None:
    metrics = pd.read_csv(output_dir / "metrics_comparison.csv")
    alignments = pd.read_csv(output_dir / "confound_alignment.csv")
    validation = json.loads((output_dir / "dataset_validation.json").read_text())
    fixed = json.loads((output_dir / "fixed_suffix_metrics.json").read_text())["metrics"]
    eot = json.loads((output_dir / "eot_metrics.json").read_text())["metrics"]
    project_root = Path(__file__).resolve().parents[1]
    historical, path_status = _historical_rows(project_root)
    for _, row in metrics.iterrows():
        historical.append({
            "experiment": f"New balanced paired {len(config['concepts'])}x50",
            "representation": row["representation"],
            "ari": float(row["ari"]),
            "matched_accuracy": float(row["matched_accuracy"]),
            "notes": "paired balanced controls",
        })
    write_csv(output_dir / "historical_comparison.csv", historical)
    write_csv(output_dir / "historical_paths.csv", path_status)

    def result_table() -> str:
        lines = ["| Representation | ARI | NMI | Matched Accuracy | Cosine Silhouette |", "|---|---:|---:|---:|---:|"]
        for _, row in metrics.iterrows():
            lines.append(f"| {row['representation']} | {row['ari']:.4f} | {row['nmi']:.4f} | {row['matched_accuracy']:.4f} | {row['cosine_silhouette']:.4f} |")
        return "\n".join(lines)

    def recall_lines(payload: dict[str, Any]) -> str:
        return ", ".join(f"{concept}={payload['per_class_recall'][concept]:.3f}" for concept in [item["name"] for item in config["concepts"]])

    alignment_lines = ["| Representation | Concept NMI | Facet NMI | Length-bin NMI | Round NMI |", "|---|---:|---:|---:|---:|"]
    for _, row in alignments.iterrows():
        alignment_lines.append(f"| {row['representation']} | {row['concept_nmi']:.4f} | {row['facet_nmi']:.4f} | {row['length_bin_nmi']:.4f} | {row['generation_round_nmi']:.4f} |")
    history_lines = ["| Experiment | Representation | ARI | Matched Accuracy | Notes |", "|---|---|---:|---:|---|"]
    for row in historical:
        history_lines.append(f"| {row['experiment']} | {row['representation']} | {row['ari']:.4f} | {row['matched_accuracy']:.4f} | {row['notes']} |")
    found = [row["path"] for row in path_status if row["status"] == "found"]
    missing = [row["path"] for row in path_status if row["status"] == "missing"]
    concepts = ", ".join(item["name"] for item in config["concepts"])
    agreement = json.loads((output_dir / "representation_agreement.json").read_text())["cluster_assignment_nmi"]
    interpretation = []
    old_diverse_fixed = next(
        (
            row for row in historical
            if row["experiment"] == "Diverse unbalanced 4x50" and row["representation"] == "fixed suffix"
        ),
        None,
    )
    balanced_fixed_ari = float(metrics.loc[metrics["representation"] == "Fixed suffix", "ari"].iloc[0])
    if old_diverse_fixed is not None and len(config["concepts"]) == 4:
        delta = balanced_fixed_ari - float(old_diverse_fixed["ari"])
        interpretation.append(
            f"Fixed-suffix ARI is {balanced_fixed_ari:.4f}, {delta:+.4f} relative to the previous "
            f"diverse unbalanced 4×50 result. This is a descriptive historical comparison, not a causal estimate."
        )
    if {"fox", "bear"}.issubset({item["name"] for item in config["concepts"]}):
        fixed_fb = (
            int(fixed["assignment_counts_by_true_class"]["fox"]["bear"])
            + int(fixed["assignment_counts_by_true_class"]["bear"]["fox"])
        )
        eot_fb = (
            int(eot["assignment_counts_by_true_class"]["fox"]["bear"])
            + int(eot["assignment_counts_by_true_class"]["bear"]["fox"])
        )
        interpretation.append(
            f"Fox↔bear confusion totals {fixed_fb} descriptions for the fixed suffix and {eot_fb} for raw EOT."
        )
    for _, row in alignments.iterrows():
        largest_confound = max(float(row["facet_nmi"]), float(row["length_bin_nmi"]), float(row["generation_round_nmi"]))
        relation = "exceeds" if float(row["concept_nmi"]) > largest_confound else "does not exceed"
        interpretation.append(
            f"For {row['representation']}, concept NMI ({row['concept_nmi']:.4f}) {relation} the largest "
            f"controlled-variable NMI ({largest_confound:.4f})."
        )
    interpretation.append(
        f"The two representations' cluster assignments have NMI {agreement:.4f}; the confusion matrices "
        "provide the class-level comparison."
    )
    report = f"""# Balanced Paired Concept-Description Clustering

## 1. Research Question

Do name-free descriptions of {concepts} form concept-dependent clusters after semantic facet, sentence length, generation slot, generation round, and generation instructions are controlled?

## 2. Manipulated and Controlled Variables

The manipulated variable is the animal concept being described. Controlled variables are the number of descriptions per concept, semantic-facet distribution, effective SD 1.4 CLIP token-length distribution, generation slot, generation round, generation source and model, candidate-generation instructions, concept-name exclusion rules, SD 1.4 generation-validation protocol, embedding model, and spherical k-means hyperparameters.

Measured outcomes are Adjusted Rand Index, Normalized Mutual Information, Hungarian-matched clustering accuracy, cosine silhouette score, per-class recall, and matched confusion matrices. True concept labels were unavailable to embedding preprocessing and clustering and were introduced only for post-hoc evaluation.

## 3. Balanced Paired Dataset Design

The dataset contains {len(config['concepts'])} concepts and 50 shared slots. Each of ten facets has five slots with the same short, short, medium, long, long schedule. Every slot fixes a common semantic instruction and generation round; one accepted description is retained for every `(concept, slot)` pair. Stored descriptions remain unsuffixed.

Short prompts contain 14–19 effective tokens, medium prompts 21–26, and long prompts 28–33, measured by the actual SD 1.4 tokenizer. The deterministic candidate source is `{config['paired_generation']['source']}` using `{config['paired_generation']['model']}` under the same generation policy for every concept. Replenishment changes only a missing concept-slot entry and is recorded separately from the controlled slot round.

## 4. Validation and Balance Checks

Intended balance constraints: **{validation['status']}** ({validation['observed_rows']}/{validation['expected_rows']} rows). The machine-readable checks and vocabulary audit are in `dataset_validation.json`; compact concept-level counts are in `dataset_balance.csv`. No accepted prompt contains a configured concept name or lexical variant, falls outside its assigned token range, or is truncated.

SD 1.4 generation validation reused the repository's original three-seed, two-stage generation settings and independent CLIP ensemble thresholds. The same seed list and image-generation settings were used for every concept in a shared slot.

## 5. Fixed-Suffix Results

The fixed suffix is exactly ` This sentence describes the concept`. It is appended only during extraction, and the final non-special token must decode to `concept`. Its contextual final-layer hidden state is extracted and row-L2-normalized.

Per-class recall: {recall_lines(fixed)}.

![Fixed-suffix confusion](confusion_fixed_suffix.png)

## 6. Unsuffixed EOT Results

The original description is tokenized without any prefix or suffix. The readout index is `attention_mask.sum(dim=1) - 1`, selecting the actual EOT token from the final SD 1.4 CLIP hidden state. Each vector is row-L2-normalized; no global centering is applied.

Per-class recall: {recall_lines(eot)}.

![EOT confusion](confusion_eot.png)

## 7. Comparison

{result_table()}

Post-hoc cluster alignment with controlled writing variables:

{chr(10).join(alignment_lines)}

Historical comparison:

{chr(10).join(history_lines)}

Historical result files found: {len(found)}. Missing: {len(missing)}. Exact paths are listed in `historical_paths.csv`.

## 8. Interpretation

{chr(10).join(f'- {line}' for line in interpretation)}

Numerical differences should be read conservatively. PCA figures are qualitative only; all clustering and reported metrics use the original 768-dimensional embeddings.

## 9. Limitations

The SD validation classifier is a closed-set CLIP ensemble rather than human ground truth. Candidate prose comes from one controlled deterministic source, so the experiment controls source variation but does not establish cross-author generality. Hungarian matching uses labels only after clustering. Spherical k-means imposes a fixed number of approximately spherical clusters and does not prove that the representation naturally contains exactly that many modes.
"""
    atomic_write_text(output_dir / "report.md", report)
    missing_outputs = [name for name in REQUESTED_OUTPUTS if not (output_dir / name).is_file()]
    empty_outputs = [
        name for name in REQUESTED_OUTPUTS
        if (output_dir / name).is_file() and (output_dir / name).stat().st_size == 0
    ]
    if missing_outputs or empty_outputs:
        raise RuntimeError(f"Requested output validation failed: missing={missing_outputs}, empty={empty_outputs}")


def run_all(config: dict[str, Any], output_dir: Path) -> None:
    generate_and_validate(config, output_dir)
    run_analysis(config, output_dir)
    build_report(config, output_dir)


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    concepts = [item["name"] for item in config["concepts"]]
    facets = [item["id"] for item in config["facets"]]
    if concepts not in (PRIMARY_CONCEPTS, EXTENSION_CONCEPTS):
        raise ValueError(f"Concept order must be primary four or configured eight, got {concepts}")
    if facets != FACETS:
        raise ValueError(f"Facet order must match the controlled ten-facet design, got {facets}")
    if config["model"]["model_id"] != "CompVis/stable-diffusion-v1-4":
        raise ValueError("This experiment must use CompVis/stable-diffusion-v1-4")
    if config["readout"]["fixed_suffix"] != FIXED_SUFFIX or config["readout"]["selected_token"] != "concept":
        raise ValueError("Fixed-suffix config differs from the repository's exact previous readout")
    if int(config["spherical_kmeans"]["k"]) != len(concepts):
        raise ValueError("spherical_kmeans.k must equal the number of configured concepts")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Balanced paired concept-description clustering")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["generate-validate", "analyze", "report", "all"]:
        item = sub.add_parser(name)
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--slots", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load_config(args.config)
    if args.command == "generate-validate":
        generate_and_validate(config, args.output)
    elif args.command == "analyze":
        run_analysis(config, args.output)
    elif args.command == "report":
        build_report(config, args.output)
    elif args.command == "all":
        run_all(config, args.output)
    elif args.command == "smoke":
        smoke_config = json.loads(json.dumps(config))
        smoke_config["model"]["num_inference_steps"] = 10
        smoke_config["model"]["generation_seeds"] = [42]
        smoke_config["classifier"]["min_top1_seeds"] = 1
        smoke_config["paired_generation"]["candidates_per_pair_per_round"] = 1
        run_smoke(smoke_config, args.output, slot_limit=args.slots)


if __name__ == "__main__":
    main()
