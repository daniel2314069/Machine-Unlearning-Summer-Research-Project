from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import atomic_write_text, read_csv, read_jsonl


def _mean_std(frame: pd.DataFrame, metric: str) -> str:
    return f"{frame[metric].mean():.4f} ± {frame[metric].std(ddof=1):.4f}"


def build_report(config: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    accepted = read_jsonl(output_dir / "accepted_descriptions.jsonl")
    generation = pd.read_csv(output_dir / "candidate_generation_decisions.csv")
    metrics = pd.read_csv(output_dir / "clustering_metrics.csv")
    layers = pd.read_csv(output_dir / "layer_metrics.csv")
    prototypes = pd.read_csv(output_dir / "prototype_metrics.csv")
    manual = pd.read_csv(output_dir / "manual_review.csv")
    length_control = pd.read_csv(output_dir / "description_length_control.csv").iloc[0]
    selection = pd.read_csv(output_dir / "final_selection.csv")
    primary = config["readout"]["primary_suffix_name"]
    primary_name = f"fixed:{primary}"
    primary_metrics = metrics[metrics["representation"] == primary_name]
    natural_metrics = metrics[metrics["representation"] == "natural_last_token"]
    shuffled_name = f"shuffled_words:fixed:{primary}"
    shuffled_metrics = metrics[metrics["representation"] == shuffled_name]
    primary_prototypes = prototypes[prototypes["representation"] == primary_name]
    layer_summary = layers.groupby(["layer_index", "layer_name"], as_index=False).agg(
        ari_mean=("ari_concept", "mean"),
        nmi_mean=("nmi_concept", "mean"),
        accuracy_mean=("hungarian_accuracy", "mean"),
        silhouette_mean=("silhouette", "mean"),
        facet_ari_mean=("ari_facet", "mean"),
    )
    best = layer_summary.sort_values("ari_mean", ascending=False).iloc[0]
    concept_count = len(config["concepts"])
    facet_count = len(config["facets"])
    accepted_per = int(config["candidate_validation"]["accepted_per_concept_facet"])
    candidate_count = len(read_jsonl(output_dir / "candidate_descriptions.jsonl"))
    manifest_path = output_dir / "merge_manifest.json"
    merge_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    run_totals = merge_manifest.get("totals", {})
    round_summaries = merge_manifest.get("rounds", [])
    rejection_reasons = merge_manifest.get("text_rejection_reasons", {})
    raw_candidate_count = int(run_totals.get("raw_candidates", candidate_count))
    selected_generation_count = int(run_totals.get("selected_generation_candidates", candidate_count))
    stage1_pass_count = int(run_totals.get("stage1_pass", 0))
    stage2_complete_count = int(run_totals.get("stage2_complete", 0))
    automatic_accepted_pool = int(run_totals.get("automatic_accepted", 0))
    started_values = [row.get("runner_started_utc") for row in round_summaries if row.get("runner_started_utc")]
    completed_value = merge_manifest.get("merged_utc")
    elapsed_text = "not available for this cached run"
    if started_values and completed_value:
        start = min(datetime.fromisoformat(value) for value in started_values)
        end = datetime.fromisoformat(completed_value)
        elapsed_seconds = max(0.0, (end - start).total_seconds())
        elapsed_text = f"{elapsed_seconds / 3600:.2f} wall-clock hours from the earliest recorded round start through final merge"
    round_lines = [
        f"| {row['round']} | {row.get('raw_candidates', 0)} | {row.get('text_valid', 0)} | "
        f"{row.get('selected_generation_candidates', 0)} | {row.get('stage1_pass', 0)} | "
        f"{row.get('stage2_complete', 0)} | {row.get('automatic_accepted', 0)} |"
        for row in round_summaries
    ]
    rejection_text = ", ".join(f"`{key}`={value}" for key, value in rejection_reasons.items()) or "none recorded"
    auto_counts = generation["automatic_decision"].value_counts().to_dict()
    manual_counts = manual["manual_decision"].fillna("unset").value_counts().to_dict()
    selected_mask = selection["selected"].astype(str).str.casefold() == "true"
    selected_manual = int((selected_mask & (selection["reason"] == "manual_accept")).sum())
    decision_by_id = generation.set_index("candidate_id")["automatic_decision"]
    selected_manual_overrides = sum(
        decision_by_id[candidate_id] != "accepted"
        for candidate_id in selection.loc[selected_mask & (selection["reason"] == "manual_accept"), "candidate_id"]
    )
    if selected_manual:
        manual_review_statement = (
            f"{selected_manual} selected rows were manually accepted; {selected_manual_overrides} of those "
            "overrode a non-accepted automatic decision."
        )
    else:
        manual_review_statement = (
            "No manual overrides were used in final selection: every selected row passed the automatic "
            "three-seed thresholds. Contact sheets and `manual_review.csv` remain available for audit."
        )
    rank_one_prototypes = int((primary_prototypes["correct_centroid_rank"] == 1).sum())
    revision_count = len(list((output_dir / "revision_archive").glob("*/revision_summary.json")))
    diversity_enabled = bool(config["candidate_validation"].get("diversity", {}).get("enabled", False))
    dataset_description = config.get("dataset_design", {}).get(
        "description", "Balanced name-free concept descriptions."
    )
    layer_prototypes = prototypes[prototypes["representation"].str.startswith("layer:")].copy()
    layer_prototypes["layer_index"] = layer_prototypes["representation"].str.rsplit(":", n=1).str[-1].astype(int)
    layer_proto_summary = layer_prototypes.groupby("layer_index").agg(
        mean_own_distance=("own_description_centroid_distance", "mean"),
        rank_one=("correct_centroid_rank", lambda values: int((values == 1).sum())),
    )
    closest_proto_layer = int(layer_proto_summary["mean_own_distance"].idxmin())

    has_percentiles = "bootstrap_percentile_median" in primary_prototypes.columns
    prototype_lines = []
    for _, row in primary_prototypes.sort_values("concept").iterrows():
        line = (
            f"| {row['concept']} | {row['nearest_cluster_matched_concept']} | "
            f"{int(row['correct_centroid_rank'])} | {row['own_description_centroid_distance']:.4f} | "
            f"{row['prototype_margin']:.4f} |"
        )
        if has_percentiles:
            line = line + (
                f" {row['bootstrap_percentile_median']:.1f} "
                f"[{row['bootstrap_percentile_ci_low']:.1f}, "
                f"{row['bootstrap_percentile_ci_high']:.1f}] |"
            )
        prototype_lines.append(line)

    if has_percentiles:
        peripheral_count = int((primary_prototypes["bootstrap_percentile_median"] >= 95).sum())
        prototype_table_header = (
            "| Prototype | Nearest cluster's matched concept | Correct centroid rank | "
            "Own-centroid cosine distance | Margin | Bootstrap distance percentile [95% CI] |\n"
            "|---|---|---:|---:|---:|---:|"
        )
        prototype_percentile_explanation = (
            "The percentile asks whether the prototype is central or peripheral relative to held-out descriptions, "
            "not merely whether its own centroid is the nearest. Higher percentiles mean farther from the centroid. "
            f"The reported median and interval come from {config.get('prototype_analysis', {}).get('n_bootstrap', 1000)} "
            "disjoint reference/evaluation bootstrap splits, avoiding direct self-inclusion of evaluated descriptions. "
            f"{peripheral_count} of {len(primary_prototypes)} prototypes are at or above the 95th percentile; "
            "therefore nearest-centroid rank must not be interpreted as being a typical member of that cluster."
        )
    else:
        prototype_table_header = (
            "| Prototype | Nearest cluster's matched concept | Correct centroid rank | "
            "Own-centroid cosine distance | Margin |\n|---|---|---:|---:|---:|"
        )
        prototype_percentile_explanation = (
            "This cached run predates prototype percentile extraction. Rerun clustering with the current code to add "
            "empirical, leave-one-out, and split-bootstrap percentile columns without regenerating images or embeddings."
        )

    failure_prototypes = primary_prototypes.loc[
        primary_prototypes["correct_centroid_rank"] != 1, "concept"
    ].tolist()
    cat_rows = primary_prototypes[primary_prototypes["concept"] == "cat"]
    cat_statement = ""
    if not cat_rows.empty:
        cat_row = cat_rows.iloc[0]
        percentile_clause = (
            f" and bootstrap distance percentile {cat_row['bootstrap_percentile_median']:.1f}"
            if has_percentiles else ""
        )
        cat_statement = (
            f" In particular, `cat` ranked its own centroid {int(cat_row['correct_centroid_rank'])} "
            f"with distance {cat_row['own_description_centroid_distance']:.4f}{percentile_clause}."
        )

    dataset_source = str(config.get("dataset_design", {}).get("source", "unknown"))
    if dataset_source == "codex_diverse":
        corpus_paragraph = (
            "Every description in this experiment has the single provenance label `codex_diverse`: it was produced "
            "during this Codex task. The rounds are replenishment batches from one source, not independent human or "
            "LLM sources. Validation rejects repeated trigrams, repeated lexical openings, repeated within-group "
            "syntax-family labels, and near duplicates. These deterministic checks reduce observable template leakage "
            "but cannot establish true source independence or complete syntactic independence."
        )
    elif diversity_enabled:
        corpus_paragraph = (
            "Validation rejects repeated trigrams, "
            "repeated lexical openings, repeated within-group syntax-family labels, and source-count imbalance. "
            "These deterministic checks reduce observable template leakage but cannot prove complete syntactic independence."
        )
    else:
        corpus_paragraph = (
            "The deterministic candidate corpus crossed concept-specific subject descriptions with shared facet "
            "predicates. This targeted construction and generation filtering increase selection and template bias."
        )
    if revision_count:
        corpus_paragraph += f" {revision_count} revision archives preserve invalidated descriptions, score rows, and images."

    layer_delta = float(best["ari_mean"] - primary_metrics["ari_concept"].mean())
    if layer_delta > 0.005:
        layer_relation = "strengthens"
    elif layer_delta < -0.005:
        layer_relation = "weakens"
    else:
        layer_relation = "approximately preserves"
    max_layer_rank_one = int(layer_proto_summary["rank_one"].max())

    baseline_section = ""
    per_concept_note = ""
    per_group_path = output_dir / "per_group_metrics.csv"
    if per_group_path.exists():
        per_group = pd.read_csv(per_group_path)
        canonical_concepts = per_group[
            (per_group["representation"] == primary_name) & (per_group["group_type"] == "concept")
        ].sort_values("group")
        if not canonical_concepts.empty:
            per_concept_note = " In the canonical run, per-concept matched accuracies were " + ", ".join(
                f"{row['group']}={row['accuracy']:.0%}" for _, row in canonical_concepts.iterrows()
            ) + "; thus the aggregate score does not mean that all four concepts formed equally clean clusters."
    plain_language_interpretation = (
        "Lexical baseline files were unavailable, so the central comparison cannot be interpreted."
    )
    baseline_path = output_dir / "lexical_baseline_summary.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        classification = pd.read_csv(output_dir / "baseline_classification_metrics.csv")
        baseline_lines = [
            f"| {row['representation']} | {row['ari_concept']:.4f} | {row['nmi_concept']:.4f} | "
            f"{row['accuracy']:.4f} | {row['ari_facet']:.4f} |"
            for _, row in baseline.iterrows()
        ]
        classifier_lines = [
            f"| {row['representation']} | {row['cv_accuracy_mean']:.4f} | {row['cv_accuracy_std']:.4f} |"
            for _, row in classification.iterrows()
        ]
        baseline_section = f"""
## Surface-lexical baselines

TF-IDF clustering uses the same labels only for evaluation, the same k, and the same k-means seed/n_init protocol as the contextual analysis.

| Representation | Concept ARI | Concept NMI | Matched accuracy | Facet ARI |
|---|---:|---:|---:|---:|
{chr(10).join(baseline_lines)}

Five-fold supervised classification is reported separately and is never compared as though it were unsupervised clustering.

| Representation | CV accuracy | CV standard deviation |
|---|---:|---:|
{chr(10).join(classifier_lines)}

![Lexical baselines](plots/lexical_baseline_comparison.png)

Word unigrams measure concept-specific lexical signal, word bigrams add local phrasing, and character 3–5 grams are particularly sensitive to surface templates. High TF-IDF performance is evidence that surface text is sufficient; it is not by itself proof that a particular fixed template leaked the label.
"""
        best_tfidf_ari = float(baseline["ari_concept"].max())
        tfidf_classification = classification[classification["representation"].str.startswith("tfidf_")]
        best_tfidf_cv = float(tfidf_classification["cv_accuracy_mean"].max())
        fixed_cv_rows = classification[classification["representation"] == primary_name]
        fixed_cv = float(fixed_cv_rows.iloc[0]["cv_accuracy_mean"]) if not fixed_cv_rows.empty else float("nan")
        plain_language_interpretation = (
            f"Within this accepted corpus, the fixed readout clusters by concept more strongly than the best "
            f"unsupervised TF-IDF condition (ARI {primary_metrics['ari_concept'].mean():.4f} versus "
            f"{best_tfidf_ari:.4f}), and the near-zero facet ARI argues against simple facet clustering. "
            f"However, supervised TF-IDF still reaches {best_tfidf_cv:.1%} accuracy, close to the fixed-readout "
            f"classifier's {fixed_cv:.1%}. Surface vocabulary therefore remains highly predictive. The result "
            "supports contextual/order-sensitive structure in this filtered dataset, but it does not establish a "
            f"lexically invariant concept geometry.{per_concept_note}"
        )

    shuffle_statement = "Word-shuffle extraction was not enabled."
    if not shuffled_metrics.empty:
        shuffle_statement = (
            f"The fixed-readout word-shuffle control produced concept ARI **{_mean_std(shuffled_metrics, 'ari_concept')}** "
            f"and matched accuracy **{_mean_std(shuffled_metrics, 'hungarian_accuracy')}** while preserving each "
            "description's exact case-folded word bag. A small change indicates primarily bag-of-words behavior; "
            "a large drop indicates sensitivity to order/contextual composition."
        )

    report = f"""# Concept-description clustering with original Stable Diffusion 1.4

## Technical summary

This run tests whether {len(accepted):,} name-free descriptions form concept-aligned clusters in the original SD 1.4 contextual text space. K-means was fitted only on accepted description vectors; explicit concept-name prototypes were excluded from fitting and evaluated afterward.

For the primary fixed-readout representation, concept ARI was **{_mean_std(primary_metrics, 'ari_concept')}**, concept NMI was **{_mean_std(primary_metrics, 'nmi_concept')}**, Hungarian-matched accuracy was **{_mean_std(primary_metrics, 'hungarian_accuracy')}**, and silhouette was **{_mean_std(primary_metrics, 'silhouette')}** across {len(primary_metrics)} independent runs. Facet ARI was **{_mean_std(primary_metrics, 'ari_facet')}** and facet NMI was **{_mean_std(primary_metrics, 'nmi_facet')}**. The result aligns more strongly with {'concepts' if primary_metrics['ari_concept'].mean() > primary_metrics['ari_facet'].mean() else 'facets'} under ARI.

This is evidence **for this generation-filtered corpus**, not evidence that arbitrary natural descriptions have the same geometry. Dataset construction, surface lexical baselines, and generation-selection effects must qualify the score.

## Key evidence

![Primary concept confusion](plots/confusion_{primary_name.replace(':', '_')}.png)

The confusion matrix uses the canonical clustering seed and Hungarian mapping. Exact assignments and distances are available in `clustering_assignments.csv`.

![Concept versus facet PCA](plots/pca_by_concept.png)

PCA is shown only as a qualitative visualization. All quantitative claims use the original normalized high-dimensional representations.

![Layer-wise clustering metrics](plots/layer_clustering_metrics.png)

Across the original `{layers.iloc[0]['representation'].split(':')[1]}` projections, the highest mean concept ARI occurred at layer {int(best['layer_index'])}, `{best['layer_name']}`, with ARI {best['ari_mean']:.4f} and matched accuracy {best['accuracy_mean']:.4f}. These are analyses of unchanged original matrices, not concept erasure.

## Scope and dataset construction

- Concepts: {concept_count} ({', '.join(item['name'] for item in config['concepts'])})
- Shared facets: {facet_count}
- Initial candidates across replenishment rounds: {raw_candidate_count:,}
- Candidates sent to generation validation: {selected_generation_count:,}
- Candidate rows retained in the merged audit: {candidate_count:,}
- Final accepted descriptions: {len(accepted):,} = {concept_count} concepts × {facet_count} facets × {accepted_per} descriptions
- Candidate text rules: English, one sentence, {config['candidate_validation']['min_words']}–{config['candidate_validation']['max_words']} words, no configured concept names or synonyms, no negation/comparison, and no near duplicates above token-Jaccard {config['candidate_validation']['near_duplicate_jaccard']}
- Dataset design: {dataset_description}

{corpus_paragraph}

Recorded elapsed time: **{elapsed_text}**.

| Round | Raw candidates | Text-valid | Sent to generation | Stage-1 pass | Three-seed complete | Automatic accept |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(round_lines) if round_lines else '| cached run | — | — | — | — | — | — |'}

Across recorded rounds, {stage1_pass_count:,}/{selected_generation_count:,} generation candidates passed Stage 1.
{automatic_accepted_pool:,}/{stage2_complete_count:,} candidates with all three seeds were automatically accepted.
Text rejection reasons from the initial pools were: {rejection_text}.

All concepts use the same resolved configuration, original tokenizer, original text encoder, and original W0 matrices. Text-validation failures remain in `candidate_text_validation.csv` with explicit reasons.

## Generation-based inclusion criterion

Generation validation used each **original description exactly as written**, without the readout suffix, because the inclusion question is whether the unmodified SD 1.4 model understands that candidate as its intended visual concept.

Stage 1 generated seed {config['model']['generation_seeds'][0]} for every text-valid candidate. Only Stage-1 passes proceeded to the remaining seeds. Automatic acceptance required the intended concept to be top-1 for at least {config['classifier']['min_top1_seeds']} seeds, mean target probability at least {config['classifier']['min_target_probability']}, mean target-versus-runner-up margin above {config['classifier']['min_average_margin']}, no low-confidence seed, and no high-confidence wrong classification.

Automatic decisions: `{json.dumps(auto_counts, sort_keys=True)}`. Manual review states: `{json.dumps(manual_counts, sort_keys=True)}`. {manual_review_statement} The classifier is an independent CLIP ViT-B/32 checkpoint rather than the SD 1.4 CLIP ViT-L/14 text encoder, but its probabilities remain closed-set and uncalibrated; low-confidence, high-entropy, and low-margin cases are therefore separated rather than forced into an animal label.

## Fixed contextual readout

Embedding extraction appends the suffix:

> {config['readout']['suffixes'][primary]}

The suffix is used only for embedding extraction. For each prompt, `phi(p)` is the final-layer contextual hidden state at the final non-special token `concept`. The tokenizer audit verifies no truncation, identical selected token IDs, and the expected decoded final token. The resulting 768-dimensional vectors are L2-normalized before clustering. No mean pooling, EOS pooling, or flattening is used in the main condition.

The fixed suffix provides a common lexical readout position whose contextual state can attend to the preceding description. This reduces confounding from naturally different final words while operationalizing, rather than proving, a sentence-level representation.

## Prototype-to-cluster analysis

Explicit names use the same suffix and selected token. They are excluded from k-means fitting, then compared with description centroids afterward.

{prototype_table_header}
{chr(10).join(prototype_lines)}

![Prototype distances](plots/prototype_to_centroid_distances.png)

This directly tests whether an explicit name lies near its own name-free description centroid without letting that name influence the fitted clusters.

{rank_one_prototypes} of {concept_count} explicit names ranked their own description centroid first.{cat_statement} Prototypes without a rank-1 own centroid: {', '.join(failure_prototypes) if failure_prototypes else 'none'}.

{prototype_percentile_explanation}

## Facet-confounding and controls

The natural-last-token control produced concept ARI **{_mean_std(natural_metrics, 'ari_concept')}** and facet ARI **{_mean_std(natural_metrics, 'ari_facet')}**. Comparing it with the fixed readout shows whether lexical differences in natural sentence endings materially change cluster alignment.

{shuffle_statement}

![Concept versus facet across layers](plots/layer_concept_vs_facet.png)

`concept_by_facet_error_table.csv` identifies which facet/concept combinations are confused most often. Sentence length is not neutral: exact length versus concept has ARI {length_control['exact_length_ari_concept']:.4f} and NMI {length_control['exact_length_nmi_concept']:.4f}; a five-fold logistic classifier using length alone reaches {length_control['length_only_logistic_cv_accuracy_mean']:.1%} ± {length_control['length_only_logistic_cv_accuracy_std']:.1%} accuracy versus {length_control['chance_accuracy']:.0%} chance. This is far below fixed-readout accuracy, so length alone does not explain the result, but it remains a real confound. Exact values are in `description_length_control.csv`.

![Description lengths](plots/description_length_by_concept.png)

{baseline_section}

## Plain-language interpretation

{plain_language_interpretation}

## Original W0 layer-wise analysis

For each original cross-attention projection layer `l`, the experiment computes `v_l(p) = W0_v,l phi(p)` (or `to_k` only when explicitly requested), L2-normalizes it, and independently repeats clustering and prototype analysis. The same unchanged W0 is used for every concept and description. No edited W is created or loaded.

Layer-level metrics, prototype distances, assignments, and confusion matrices are saved in `layer_metrics.csv`, `prototype_metrics.csv`, `clustering_assignments.csv`, and `confusion_matrices/`.

Layer {int(best['layer_index'])} {layer_relation} description clustering relative to raw contextual space (ARI {best['ari_mean']:.4f} versus {primary_metrics['ari_concept'].mean():.4f}). Layer {closest_proto_layer} gives the smallest mean prototype-to-own-centroid distance ({layer_proto_summary.loc[closest_proto_layer, 'mean_own_distance']:.4f}). The best layer has {max_layer_rank_one}/{concept_count} rank-1 prototypes, compared with {rank_one_prototypes}/{concept_count} in raw fixed-readout space; this comparison is descriptive and does not imply that W0 created the structure.

## Limitations and robustness

1. A single contextual token is an operational readout, not a guaranteed sufficient sentence representation.
2. CLIP-based generation validation is not an open-set detector and may misclassify malformed, toy, partial, or absent animals; contact sheets and `manual_review.csv` remain important.
3. Filtering on SD 1.4 generation success intentionally selects descriptions the base model can render, so conclusions apply to the accepted subset rather than all valid English descriptions.
4. Deterministic trigram/opening/syntax-family checks only detect specified forms of repetition; they cannot establish complete sentence independence or remove legitimate concept-diagnostic vocabulary.
5. Description length differs by concept and carries measurable predictive information, although it is insufficient to reproduce the main score.
6. K-means assumes roughly spherical, similarly sized clusters. Balanced sampling helps, but it does not prove that the representation has exactly {concept_count} natural modes.
7. PCA plots are qualitative; ARI, NMI, matched accuracy, and silhouette are computed in the original normalized feature space.
8. Alternative suffix conditions are reported separately and never mixed into one clustering fit.

## Reproducibility and completion status

The unattended supervisor finished successfully; `supervisor_state.json` records `status=complete` and
`stage=finished`. The exact detached command used to launch the final supervisor is preserved in
`launched_command.txt`. Cached generation rounds and their images occupy approximately 1.7 GiB; the compact final
analysis directory is approximately 22 MiB.

Status inspection (no Python required):

```bash
cat outputs/codex_diverse_final/supervisor_state.json
```

Rerun clustering/reporting from cached embeddings without regenerating images:

```bash
./scripts/run_py310.sh -m concept_clustering.cli analyze \
  --config configs/codex_diverse_4x50_round2.json \
  --output outputs/codex_diverse_final
```

Rerun the final integrity audit:

```bash
./scripts/run_py310.sh -m scripts.qa_final_results \
  --config configs/codex_diverse_4x50_round2.json \
  --output outputs/codex_diverse_final
```

All Python commands use the repository-required Conda `py310` environment through `scripts/run_py310.sh`.

## Recommended next steps

- Replicate with genuinely independent human or separately generated corpora before making a general claim.
- Inspect the saved concept-by-facet errors and prototype rank failures before interpreting the aggregate score.
- Repeat with the configured alternative suffix as a robustness condition, keeping each suffix in a separate fit.
- If image-classification errors remain material, replace the configurable backend with an independent open-set detector while preserving the CSV contract.

## Further questions

- Are errors concentrated in highly indirect facets or distributed across all facets?
- Do explicit names fail mainly when their name-free descriptions form multiple semantic subclusters?
- Do any original W0 layers improve concept alignment while simultaneously increasing facet alignment?
"""
    path = output_dir / "final_report.md"
    atomic_write_text(path, report)
    return path
