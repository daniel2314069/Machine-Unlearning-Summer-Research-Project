#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"
require "json"

ROOT = File.expand_path(__dir__)
GENERATED_AT = "2026-08-19T07:30:48.012777+00:00"
TITLE = "Sequential OCE Pair-Erasure: Previous-Target Retain Results"

def rows(name)
  CSV.read(File.join(ROOT, name), headers: true).map(&:to_h)
end

def number(value)
  Float(value)
end

summary = rows("summary.csv")
stage1 = rows("stage1_paper_metrics.csv")
order_effects_source = rows("order_effects.csv")

order_results = summary.each_with_index.map do |row, index|
  {
    "schedule_index" => index + 1,
    "pair" => row.fetch("pair"),
    "order" => row.fetch("order"),
    "first_target" => row.fetch("first_target"),
    "second_target" => row.fetch("second_target"),
    "first_after_first" => number(row.fetch("first_target_accuracy_after_first_edit")),
    "first_after_baseline" => number(row.fetch("first_target_accuracy_after_second_baseline")),
    "first_after_retain" => number(row.fetch("first_target_accuracy_after_retain_previous_second")),
    "baseline_delta_pp" => 100.0 * number(row.fetch("first_target_raw_delta_after_second_baseline")),
    "retain_delta_pp" => 100.0 * number(row.fetch("first_target_raw_delta_after_retain_previous_second")),
    "retain_minus_baseline_first_pp" => 100.0 * (
      number(row.fetch("first_target_accuracy_after_retain_previous_second")) -
      number(row.fetch("first_target_accuracy_after_second_baseline"))
    ),
    "second_baseline" => number(row.fetch("second_target_final_accuracy_baseline")),
    "second_retain" => number(row.fetch("second_target_final_accuracy_retain_previous")),
    "retain_minus_baseline_second_pp" => 100.0 * number(row.fetch("second_target_raw_difference_retain_minus_baseline")),
    "remaining_8_baseline" => number(row.fetch("remaining_8_mean_accuracy_baseline")),
    "remaining_8_retain" => number(row.fetch("remaining_8_mean_accuracy_retain_previous")),
    "retain_minus_baseline_remaining_8_pp" => 100.0 * number(row.fetch("remaining_8_mean_raw_difference_retain_minus_baseline")),
    "stage1_acc_e" => number(row.fetch("stage1_Acc_e")),
    "stage1_acc_s" => number(row.fetch("stage1_Acc_s")),
    "stage1_h_o" => number(row.fetch("stage1_H_o"))
  }
end

stage1_results = stage1.each_with_index.map do |row, index|
  {
    "schedule_index" => index + 1,
    "pair" => row.fetch("pair"),
    "order" => row.fetch("order"),
    "target" => row.fetch("target"),
    "acc_e" => number(row.fetch("Acc_e")),
    "acc_s" => number(row.fetch("Acc_s")),
    "h_o" => number(row.fetch("H_o")),
    "samples_per_class" => 200
  }
end

first_target_deltas = order_results.flat_map do |row|
  [
    {
      "schedule_index" => row.fetch("schedule_index"),
      "pair" => row.fetch("pair"),
      "order" => row.fetch("order"),
      "first_target" => row.fetch("first_target"),
      "condition" => "Normal second edit",
      "delta_pp" => row.fetch("baseline_delta_pp"),
      "stage1_accuracy" => row.fetch("first_after_first"),
      "final_accuracy" => row.fetch("first_after_baseline"),
      "samples_per_class" => 200
    },
    {
      "schedule_index" => row.fetch("schedule_index"),
      "pair" => row.fetch("pair"),
      "order" => row.fetch("order"),
      "first_target" => row.fetch("first_target"),
      "condition" => "Retain previous target",
      "delta_pp" => row.fetch("retain_delta_pp"),
      "stage1_accuracy" => row.fetch("first_after_first"),
      "final_accuracy" => row.fetch("first_after_retain"),
      "samples_per_class" => 200
    }
  ]
end

retain_tradeoffs = order_results.flat_map do |row|
  [
    {
      "schedule_index" => row.fetch("schedule_index"),
      "pair" => row.fetch("pair"),
      "order" => row.fetch("order"),
      "outcome" => "First target final accuracy",
      "retain_minus_baseline_pp" => row.fetch("retain_minus_baseline_first_pp"),
      "baseline_accuracy" => row.fetch("first_after_baseline"),
      "retain_accuracy" => row.fetch("first_after_retain"),
      "samples_per_class" => 200
    },
    {
      "schedule_index" => row.fetch("schedule_index"),
      "pair" => row.fetch("pair"),
      "order" => row.fetch("order"),
      "outcome" => "Second target final accuracy",
      "retain_minus_baseline_pp" => row.fetch("retain_minus_baseline_second_pp"),
      "baseline_accuracy" => row.fetch("second_baseline"),
      "retain_accuracy" => row.fetch("second_retain"),
      "samples_per_class" => 200
    },
    {
      "schedule_index" => row.fetch("schedule_index"),
      "pair" => row.fetch("pair"),
      "order" => row.fetch("order"),
      "outcome" => "Remaining-eight mean accuracy",
      "retain_minus_baseline_pp" => row.fetch("retain_minus_baseline_remaining_8_pp"),
      "baseline_accuracy" => row.fetch("remaining_8_baseline"),
      "retain_accuracy" => row.fetch("remaining_8_retain"),
      "samples_per_class" => 1_600
    }
  ]
end

order_effects = order_effects_source.map.with_index do |row, index|
  {
    "row_index" => index + 1,
    "pair" => row.fetch("pair"),
    "concept" => row.fetch("concept"),
    "variant" => row.fetch("variant") == "baseline" ? "Normal sequential" : "Retain previous target",
    "accuracy_when_erased_first" => number(row.fetch("accuracy_when_erased_first")),
    "accuracy_when_erased_second" => number(row.fetch("accuracy_when_erased_second")),
    "first_minus_second_pp" => 100.0 * number(row.fetch("raw_difference_first_minus_second")),
    "samples_per_condition" => 200
  }
end

sources = [
  {
    "id" => "summary-source",
    "label" => "Sequential OCE order summary",
    "path" => "summary.csv",
    "query" => {
      "description" => "Runner-produced per-order target and remaining-eight accuracies.",
      "language" => "csv",
      "executed_at" => GENERATED_AT,
      "filters" => ["Five preregistered unordered pairs", "Both orders", "200 images per class", "Seeds 42 through 241"],
      "metric_definitions" => [
        "Target accuracy is CLIP ten-class classification accuracy; lower is better for an erased target.",
        "Remaining-eight mean is the unweighted mean accuracy over the eight classes outside the ordered target pair.",
        "Raw differences are direct percentage-point differences after multiplying stored fractional accuracies by 100."
      ]
    }
  },
  {
    "id" => "stage1-source",
    "label" => "Stage-1 paper metrics",
    "path" => "stage1_paper_metrics.csv",
    "query" => {
      "description" => "Runner-produced Acc_e, Acc_s, and H_o for every standard single-concept first edit.",
      "language" => "csv",
      "executed_at" => GENERATED_AT,
      "filters" => ["200 images per CIFAR-10 class", "Seeds 42 through 241"]
    }
  },
  {
    "id" => "per-class-source",
    "label" => "All per-class evaluation results",
    "path" => "per_class_results.csv",
    "query" => {
      "description" => "Runner-produced raw accuracy for every class-level evaluator cell, including the Original baseline.",
      "language" => "csv",
      "executed_at" => GENERATED_AT,
      "filters" => ["All ten CIFAR-10 classes", "200 images per class", "Seeds 42 through 241"]
    }
  },
  {
    "id" => "order-effects-source",
    "label" => "Recomputed order comparisons",
    "path" => "order_effects.csv",
    "query" => {
      "description" => "Final accuracy for each concept when erased first versus second, recomputed from saved summary rows.",
      "language" => "ruby",
      "executed_at" => GENERATED_AT,
      "metric_definitions" => ["First-minus-second is the direct final-accuracy difference in percentage points; no new score or significance test is applied."]
    }
  },
  {
    "id" => "audit-source",
    "label" => "Independent raw-prediction audit",
    "path" => "independent_audit.json",
    "query" => {
      "description" => "Independent standard-library Ruby recomputation from all_predictions.csv and per-cell completion artifacts.",
      "language" => "ruby",
      "executed_at" => GENERATED_AT,
      "filters" => ["All 62,000 saved prediction rows", "All 310 evaluator cells"]
    }
  },
  {
    "id" => "manifest-source",
    "label" => "Completed experiment manifest",
    "path" => "run_manifest.json",
    "query" => {
      "description" => "Protocol, environment, completion status, and artifact paths saved by the runner.",
      "language" => "json",
      "executed_at" => GENERATED_AT
    }
  },
  {
    "id" => "qualitative-source",
    "label" => "Fixed qualitative sample manifest",
    "path" => "qualitative_manifest.json",
    "query" => {
      "description" => "Precommitted qualitative seeds and the inventory of 140 raw samples and 20 contact sheets.",
      "language" => "json",
      "executed_at" => GENERATED_AT,
      "filters" => ["Seeds 42 and 43", "All five pairs", "Both orders", "Both target concepts"]
    }
  }
]

charts = [
  {
    "id" => "stage1-erasure-chart",
    "title" => "Stage-1 target accuracy by ordered sequence",
    "subtitle" => "200 images per class; lower Acc_e means stronger erasure",
    "showDescription" => true,
    "intent" => "comparison",
    "question" => "Did each ordered sequence begin with a successfully erased first target?",
    "rationale" => "A horizontal bar comparison exposes the two large Stage-1 failures without hiding the eight low-accuracy outcomes.",
    "comparisonContext" => { "unit" => "accuracy", "grain" => "ordered sequence", "denominator" => "200 generated images per target class" },
    "type" => "horizontalBar",
    "dataset" => "stage1_results",
    "sourceId" => "stage1-source",
    "encodings" => {
      "x" => { "field" => "order", "type" => "nominal", "label" => "Order" },
      "y" => { "field" => "acc_e", "type" => "quantitative", "format" => "percent", "label" => "Stage-1 target accuracy" },
      "tooltip" => [
        { "field" => "target", "type" => "text", "label" => "First target" },
        { "field" => "acc_s", "type" => "quantitative", "format" => "percent", "label" => "Acc_s" },
        { "field" => "h_o", "type" => "quantitative", "format" => "percent", "label" => "H_o" },
        { "field" => "samples_per_class", "type" => "quantitative", "format" => "number", "label" => "Images" }
      ]
    },
    "valueFormat" => "percent",
    "layout" => "full",
    "palette" => { "kind" => "sequential", "name" => "blue" },
    "legend" => { "position" => "bottom", "interactive" => false },
    "labels" => { "values" => "auto" }
  },
  {
    "id" => "first-target-delta-chart",
    "title" => "First-target accuracy change after the second edit",
    "subtitle" => "Final minus immediate Stage-1 accuracy, in percentage points; positive means erasure weakened",
    "showDescription" => true,
    "intent" => "comparison",
    "question" => "Does the second edit raise the first erased target, and does retaining it change that outcome?",
    "rationale" => "Grouped bars preserve every preregistered order and directly compare the two allowed Stage-2 conditions on the requested raw difference.",
    "comparisonContext" => { "baseline" => "accuracy immediately after Stage 1", "unit" => "percentage points", "grain" => "ordered sequence" },
    "type" => "bar",
    "dataset" => "first_target_deltas",
    "sourceId" => "summary-source",
    "encodings" => {
      "x" => { "field" => "order", "type" => "nominal", "label" => "Order" },
      "y" => { "field" => "delta_pp", "type" => "quantitative", "format" => "number", "label" => "Accuracy change" },
      "color" => { "field" => "condition", "type" => "nominal", "label" => "Second edit" },
      "tooltip" => [
        { "field" => "first_target", "type" => "text", "label" => "First target" },
        { "field" => "stage1_accuracy", "type" => "quantitative", "format" => "percent", "label" => "Stage-1 accuracy" },
        { "field" => "final_accuracy", "type" => "quantitative", "format" => "percent", "label" => "Final accuracy" },
        { "field" => "samples_per_class", "type" => "quantitative", "format" => "number", "label" => "Images" }
      ]
    },
    "valueFormat" => "number",
    "unit" => "pp",
    "layout" => "full",
    "palette" => { "kind" => "categorical", "name" => "blue-orange" },
    "legend" => { "position" => "bottom", "interactive" => false, "title" => "Second edit" },
    "labels" => { "values" => "auto" },
    "referenceLines" => [{ "axis" => "y", "value" => 0, "label" => "No change", "color" => "neutral", "lineStyle" => "solid" }]
  },
  {
    "id" => "order-difference-chart",
    "title" => "Final target accuracy when erased first versus second",
    "subtitle" => "First-position minus second-position accuracy, in percentage points; values are descriptive raw differences",
    "showDescription" => true,
    "intent" => "comparison",
    "question" => "How much does final image-level erasure differ between the two orders for each concept?",
    "rationale" => "A grouped signed bar makes the direction and magnitude of each concept's raw order difference visible for both Stage-2 variants.",
    "comparisonContext" => { "baseline" => "same concept erased in the opposite position", "unit" => "percentage points", "grain" => "concept and Stage-2 variant" },
    "type" => "bar",
    "dataset" => "order_effects",
    "sourceId" => "order-effects-source",
    "encodings" => {
      "x" => { "field" => "concept", "type" => "nominal", "label" => "Concept" },
      "y" => { "field" => "first_minus_second_pp", "type" => "quantitative", "format" => "number", "label" => "First minus second" },
      "color" => { "field" => "variant", "type" => "nominal", "label" => "Stage-2 variant" },
      "tooltip" => [
        { "field" => "pair", "type" => "text", "label" => "Pair" },
        { "field" => "accuracy_when_erased_first", "type" => "quantitative", "format" => "percent", "label" => "Erased first" },
        { "field" => "accuracy_when_erased_second", "type" => "quantitative", "format" => "percent", "label" => "Erased second" },
        { "field" => "samples_per_condition", "type" => "quantitative", "format" => "number", "label" => "Images per condition" }
      ]
    },
    "valueFormat" => "number",
    "unit" => "pp",
    "layout" => "full",
    "palette" => { "kind" => "categorical", "name" => "blue-orange" },
    "legend" => { "position" => "bottom", "interactive" => false, "title" => "Stage-2 variant" },
    "labels" => { "values" => "auto" },
    "referenceLines" => [{ "axis" => "y", "value" => 0, "label" => "No order difference", "color" => "neutral", "lineStyle" => "solid" }]
  }
]

tables = [
  {
    "id" => "order-summary-table",
    "title" => "Per-order target and preservation results",
    "subtitle" => "All ten preregistered orders; accuracies use 200 images per class",
    "showDescription" => true,
    "dataset" => "order_results",
    "defaultSort" => { "field" => "schedule_index", "direction" => "asc" },
    "density" => "dense",
    "sourceId" => "summary-source",
    "layout" => "full",
    "columns" => [
      { "field" => "schedule_index", "label" => "#", "format" => "number" },
      { "field" => "order", "label" => "Order", "type" => "text" },
      { "field" => "first_after_first", "label" => "First after Stage 1", "format" => "percent" },
      { "field" => "first_after_baseline", "label" => "First after normal Stage 2", "format" => "percent" },
      { "field" => "first_after_retain", "label" => "First after retain Stage 2", "format" => "percent" },
      { "field" => "second_baseline", "label" => "Second normal", "format" => "percent" },
      { "field" => "second_retain", "label" => "Second retain", "format" => "percent" },
      { "field" => "remaining_8_baseline", "label" => "Remaining 8 normal", "format" => "percent" },
      { "field" => "remaining_8_retain", "label" => "Remaining 8 retain", "format" => "percent" }
    ]
  },
  {
    "id" => "retain-tradeoff-table",
    "title" => "Effect of retain-previous relative to normal Stage 2",
    "subtitle" => "Retain-previous minus normal Stage-2 accuracy, in percentage points; lower target values are better",
    "showDescription" => true,
    "dataset" => "order_results",
    "defaultSort" => { "field" => "schedule_index", "direction" => "asc" },
    "density" => "spacious",
    "sourceId" => "summary-source",
    "layout" => "full",
    "columns" => [
      { "field" => "schedule_index", "label" => "#", "format" => "number" },
      { "field" => "order", "label" => "Order", "type" => "text" },
      { "field" => "retain_minus_baseline_first_pp", "label" => "First target", "format" => "number", "movement" => true },
      { "field" => "retain_minus_baseline_second_pp", "label" => "Second target", "format" => "number", "movement" => true },
      { "field" => "retain_minus_baseline_remaining_8_pp", "label" => "Remaining-eight mean", "format" => "number", "movement" => true }
    ]
  },
  {
    "id" => "order-effects-table",
    "title" => "Raw final-accuracy differences between erasure positions",
    "subtitle" => "Every concept under both Stage-2 variants; 200 images per displayed accuracy",
    "showDescription" => true,
    "dataset" => "order_effects",
    "defaultSort" => { "field" => "row_index", "direction" => "asc" },
    "density" => "dense",
    "sourceId" => "order-effects-source",
    "layout" => "full",
    "columns" => [
      { "field" => "pair", "label" => "Pair", "type" => "text" },
      { "field" => "concept", "label" => "Concept", "type" => "text" },
      { "field" => "variant", "label" => "Variant", "type" => "text" },
      { "field" => "accuracy_when_erased_first", "label" => "Erased first", "format" => "percent" },
      { "field" => "accuracy_when_erased_second", "label" => "Erased second", "format" => "percent" },
      { "field" => "first_minus_second_pp", "label" => "First minus second (pp)", "format" => "number", "movement" => true }
    ]
  }
]

blocks = [
  { "id" => "title", "type" => "markdown", "body" => "# #{TITLE}", "layout" => "full" },
  {
    "id" => "technical-summary",
    "type" => "markdown",
    "sourceId" => "summary-source",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Technical summary

      **The second OCE edit sometimes weakens the first erasure, but the effect is not universal.** Under normal sequential OCE, first-target accuracy increased in 3 of 10 orders, was unchanged in 4, and decreased in 3. The mean raw change was **+0.60 percentage points (pp)**, the median was **0 pp**, and the largest rebound was **+8.5 pp** for `bird→automobile`.

      **Order has a real descriptive image-level effect for some concepts, not all.** Under the normal sequential condition, the largest first-versus-second final-accuracy differences were **10.0 pp for bird** and **7.5 pp for automobile**. `deer↔dog` was effectively order-invariant, while `frog↔ship` showed 2.5 pp differences for both concepts.

      **Adding the previous target to the second edit's retain set helps selectively, not reliably.** Relative to normal Stage 2, first-target final accuracy was lower in 5 orders, tied in 2, and higher in 3. The mean change was **−1.05 pp** and the median was **−0.50 pp**, dominated by an **−8.5 pp** improvement for `bird→automobile`.

      **The modification has a small but non-zero second-target tradeoff and essentially no aggregate remaining-eight cost.** Second-target accuracy changed by **+0.35 pp on average** (higher is worse for erasure): 5 orders worsened, 4 tied, and 1 improved. Remaining-eight mean accuracy changed by **−0.019 pp on average**, with 7 exact ties and a range from −0.125 to +0.063 pp.
    MARKDOWN
  },
  {
    "id" => "original-section",
    "type" => "markdown",
    "sourceId" => "per-class-source",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Original baseline is effectively at the classifier ceiling

      Original SD v1.4 reached **100% accuracy on eight classes** and **99.5% on automobile and truck**, for a ten-class mean of **99.9%**. The low edited-target accuracies therefore do not come from a weak Original classifier baseline.
    MARKDOWN
  },
  {
    "id" => "stage1-section",
    "type" => "markdown",
    "sourceId" => "stage1-source",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Two Stage-1 failures limit the intended retention interpretation

      Eight first edits reached target accuracy at or below 7.5%. The exceptions were **automobile at 90.0%** and **bird at 57.0%**. Therefore, the `automobile↔bird` sequences remain part of the fixed primary schedule, but they do not begin from a cleanly erased first target. The strongest apparent benefit of retain-previous occurs for `bird→automobile`, so it should not be treated as decisive evidence for preserving a successful first erasure.
    MARKDOWN
  },
  { "id" => "stage1-chart-block", "type" => "chart", "chartId" => "stage1-erasure-chart", "layout" => "full" },
  {
    "id" => "sequential-section",
    "type" => "markdown",
    "sourceId" => "summary-source",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Normal Stage 2 causes pair- and order-specific first-target changes

      A positive bar means the first erased class became more classifiable after the second edit. The clearest rebound is `bird→automobile` (+8.5 pp), followed by `airplane→cat` (+3.0 pp) and `ship→frog` (+0.5 pp). Four orders are unchanged, while `frog→ship`, `horse→truck`, and `truck→horse` each improve by 2.0 pp. This supports a **sometimes**, not **generally**, answer to whether the second edit breaks the first.
    MARKDOWN
  },
  { "id" => "first-delta-chart-block", "type" => "chart", "chartId" => "first-target-delta-chart", "layout" => "full" },
  {
    "id" => "retain-section",
    "type" => "markdown",
    "sourceId" => "summary-source",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Retain-previous improves five orders but is not a uniform safeguard

      Compared directly with normal Stage 2, retain-previous lowers first-target final accuracy in `airplane→cat` (−1.0 pp), `bird→automobile` (−8.5 pp), `frog→ship` (−1.0 pp), `ship→frog` (−1.5 pp), and `truck→horse` (−1.0 pp). It ties twice and worsens `automobile→bird` (+1.5 pp), `deer→dog` (+0.5 pp), and `horse→truck` (+0.5 pp).

      The second target is not blocked wholesale, but the modification is not free: five orders increase second-target accuracy by 0.5–1.5 pp, four tie, and `frog→ship` improves by 2.0 pp. The remaining-eight means stay between 99.81% and 100% across both variants.
    MARKDOWN
  },
  { "id" => "retain-table-block", "type" => "table", "tableId" => "retain-tradeoff-table", "layout" => "full" },
  {
    "id" => "order-section",
    "type" => "markdown",
    "sourceId" => "order-effects-source",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Order differences concentrate in specific pairs

      The chart compares each concept's final accuracy when erased first with its final accuracy when erased second. Normal sequential OCE differs by 10.0 pp for bird, 7.5 pp for automobile, 2.5 pp for airplane, frog, and ship, and at most 0.5 pp for the other five concepts. Retain-previous narrows the bird difference to 1.0 pp but leaves a 6.0 pp automobile difference and a 4.5 pp frog difference. These are descriptive raw differences, not a newly defined score or a statistical significance claim.
    MARKDOWN
  },
  { "id" => "order-chart-block", "type" => "chart", "chartId" => "order-difference-chart", "layout" => "full" },
  { "id" => "order-table-block", "type" => "table", "tableId" => "order-effects-table", "layout" => "full" },
  {
    "id" => "exact-results-section",
    "type" => "markdown",
    "sourceId" => "summary-source",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Exact per-order results preserve the requested raw metrics

      The table reports the first target immediately after Stage 1, both final first-target accuracies, both final second-target accuracies, and both remaining-eight means. No multi-target harmonic score or additional geometry diagnostic is introduced.
    MARKDOWN
  },
  { "id" => "summary-table-block", "type" => "table", "tableId" => "order-summary-table", "layout" => "full" },
  {
    "id" => "scope-section",
    "type" => "markdown",
    "sourceId" => "manifest-source",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Scope, data, and metric definitions

      The run used Stable Diffusion v1.4 and the repository's CIFAR-10 OCE object protocol: `a photo of the {class}`, 200 images per class, seeds 42–241 shared across all conditions, PNDM, 50 inference steps, CFG 7.5, 512×512 resolution, and `openai/clip-vit-base-patch32` ten-class argmax classification.

      Target accuracy is lower-is-better. First-target change is final accuracy after Stage 2 minus accuracy immediately after Stage 1. Second-target comparison and remaining-eight comparison are retain-previous minus normal Stage 2. Every percentage-point statement is a direct subtraction of raw accuracies.
    MARKDOWN
  },
  {
    "id" => "method-section",
    "type" => "markdown",
    "sourceId" => "audit-source",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Independent validation reproduces every saved aggregate

      The audit re-read all **62,000 prediction rows** without invoking a model. It found **310 complete 200-image cells**, no duplicate `(group, checkpoint, concept, sample_index, seed)` keys, exact seed coverage, matching prompts and labels, and maximum CLIP probability-sum error below `2.4×10⁻⁷`. It independently reproduced all 310 per-class rows, all 10 summary rows, all Stage-1 paper metrics, every per-cell prediction/metrics hash, and the shared Stage-1 parent recorded by both Stage-2 variants.
    MARKDOWN
  },
  {
    "id" => "limitations-section",
    "type" => "markdown",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Limitations and robustness boundaries

      - The evidence is descriptive for this fixed five-pair schedule; no inferential test or generalization beyond these concepts is claimed.
      - Accuracies move in 0.5 pp increments because each class has 200 images.
      - `automobile↔bird` is retained in the primary result as preregistered, but its Stage-1 failures weaken the causal interpretation of first-erasure retention.
      - Fixed qualitative seeds 42 and 43 cover every pair, order, and target without cherry-picking. They support visual inspection but do not replace the 200-image CLIP evaluator.
      - Formal images were deleted only after successful evaluation; raw predictions, manifests, hashes, and the separately packaged qualitative samples remain saved.
    MARKDOWN
  },
  {
    "id" => "recommendation-section",
    "type" => "markdown",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Recommended next step

      Treat retain-previous as an **optional, pair-dependent safeguard**, not a default improvement. For a follow-up confirmation, rerun the same fixed protocol with additional independently fixed seed sets or repetitions—without changing pairs or hyperparameters—and predeclare that the key check is whether the five observed first-target improvements recur without a larger second-target penalty.
    MARKDOWN
  },
  {
    "id" => "questions-section",
    "type" => "markdown",
    "layout" => "full",
    "body" => <<~MARKDOWN.strip
      ## Further questions

      - Does the `bird→automobile` retention improvement persist when the first bird erasure is itself successful?
      - Are the 0.5–1.5 pp second-target penalties stable across independent seed sets?
      - Which of the observed order differences replicate without modifying the fixed pair schedule?
    MARKDOWN
  }
]

artifact = {
  "surface" => "report",
  "manifest" => {
    "version" => 1,
    "surface" => "report",
    "title" => TITLE,
    "description" => "Validated image-level results for ten ordered two-concept OCE erasure sequences and the previous-target retain modification.",
    "generatedAt" => GENERATED_AT,
    "sources" => sources,
    "charts" => charts,
    "tables" => tables,
    "blocks" => blocks
  },
  "snapshot" => {
    "version" => 1,
    "generatedAt" => GENERATED_AT,
    "status" => "ready",
    "datasets" => {
      "order_results" => order_results,
      "stage1_results" => stage1_results,
      "first_target_deltas" => first_target_deltas,
      "retain_tradeoffs" => retain_tradeoffs,
      "order_effects" => order_effects
    }
  },
  "sources" => sources
}

File.write(File.join(ROOT, "artifact.json"), JSON.pretty_generate(artifact) + "\n")
puts "Wrote #{File.join(ROOT, 'artifact.json')}"
