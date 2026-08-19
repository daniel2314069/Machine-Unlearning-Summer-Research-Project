#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"
require "digest"
require "fileutils"
require "json"
require "set"

CLASSES = %w[airplane automobile bird cat deer dog frog horse ship truck].freeze
EXPECTED_SEEDS = (42..241).to_a.freeze
TOLERANCE = 1e-12

def fail_audit(message)
  raise RuntimeError, message
end

def close_enough?(left, right, tolerance = TOLERANCE)
  (left.to_f - right.to_f).abs <= tolerance
end

def read_csv(path)
  CSV.read(path, headers: true).map(&:to_h)
end

def write_csv(path, headers, rows)
  CSV.open(path, "w", write_headers: true, headers: headers, row_sep: "\n") do |csv|
    rows.each { |row| csv << headers.map { |header| row.fetch(header) } }
  end
end

result_root = File.expand_path(ARGV.fetch(0))
output_dir = File.expand_path(ARGV.fetch(1, File.join(result_root, "independent_audit")))
FileUtils.mkdir_p(output_dir)

predictions_path = File.join(result_root, "raw", "all_predictions.csv")
per_class_path = File.join(result_root, "per_class_results.csv")
summary_path = File.join(result_root, "summary.csv")
stage1_path = File.join(result_root, "stage1_paper_metrics.csv")
schedule_path = File.join(result_root, "inputs", "pair_schedule.csv")
validation_path = File.join(result_root, "final_validation.json")
manifest_path = File.join(result_root, "run_manifest.json")

[predictions_path, per_class_path, summary_path, stage1_path, schedule_path,
 validation_path, manifest_path].each do |path|
  fail_audit("Missing required source file: #{path}") unless File.file?(path)
end

aggregates = Hash.new do |hash, key|
  hash[key] = { "n" => 0, "correct" => 0, "seeds" => Set.new, "samples" => Set.new }
end
seen_records = Set.new
prediction_rows = 0
maximum_probability_sum_error = 0.0

CSV.foreach(predictions_path, headers: true) do |row|
  prediction_rows += 1
  cell_key = [row.fetch("group"), row.fetch("checkpoint"), row.fetch("concept")]
  record_key = cell_key + [row.fetch("sample_index"), row.fetch("seed")]
  fail_audit("Duplicate prediction key: #{record_key.inspect}") unless seen_records.add?(record_key)

  concept = row.fetch("concept")
  sample_index = Integer(row.fetch("sample_index"))
  seed = Integer(row.fetch("seed"))
  fail_audit("Unknown concept: #{concept}") unless CLASSES.include?(concept)
  fail_audit("Expected label mismatch at row #{prediction_rows}") unless row.fetch("expected_label") == concept
  fail_audit("Prompt mismatch at row #{prediction_rows}") unless row.fetch("prompt") == "a photo of the #{concept}"
  fail_audit("Seed/index mismatch at row #{prediction_rows}") unless seed == sample_index + 42
  fail_audit("Invalid correctness value at row #{prediction_rows}") unless %w[True False].include?(row.fetch("correct"))

  probability_sum = CLASSES.sum { |label| Float(row.fetch("prob_#{label}")) }
  maximum_probability_sum_error = [maximum_probability_sum_error, (probability_sum - 1.0).abs].max

  aggregate = aggregates[cell_key]
  aggregate["n"] += 1
  aggregate["correct"] += 1 if row.fetch("correct") == "True"
  aggregate["seeds"].add(seed)
  aggregate["samples"].add(sample_index)
end

fail_audit("Expected 62,000 predictions, found #{prediction_rows}") unless prediction_rows == 62_000
fail_audit("Expected 310 evaluator cells, found #{aggregates.length}") unless aggregates.length == 310
fail_audit("Probability rows do not sum to one") unless maximum_probability_sum_error < 1e-5

aggregates.each do |key, aggregate|
  fail_audit("Cell #{key.inspect} has #{aggregate['n']} predictions") unless aggregate["n"] == 200
  fail_audit("Cell #{key.inspect} has the wrong seeds") unless aggregate["seeds"].to_a.sort == EXPECTED_SEEDS
  fail_audit("Cell #{key.inspect} has the wrong sample indices") unless aggregate["samples"].to_a.sort == (0..199).to_a
end

accuracy = lambda do |group, checkpoint, concept|
  aggregate = aggregates.fetch([group, checkpoint, concept])
  aggregate.fetch("correct").fdiv(aggregate.fetch("n"))
end

per_class_rows = read_csv(per_class_path)
fail_audit("Expected 310 per-class rows, found #{per_class_rows.length}") unless per_class_rows.length == 310
per_class_rows.each do |row|
  if row.fetch("variant") == "original"
    calculated = accuracy.call("original", "original", row.fetch("class"))
  else
    group = row.fetch("order").sub("->", "_then_")
    calculated = accuracy.call(group, row.fetch("variant"), row.fetch("class"))
  end
  fail_audit("Per-class mismatch: #{row.inspect}") unless close_enough?(calculated, row.fetch("accuracy"))
  fail_audit("Per-class sample count mismatch: #{row.inspect}") unless Integer(row.fetch("number_of_generated_samples")) == 200
end

schedule = read_csv(schedule_path)
fail_audit("Expected 10 ordered pairs, found #{schedule.length}") unless schedule.length == 10
fail_audit("Expected each class once as first target") unless schedule.map { |row| row.fetch("first_target") }.sort == CLASSES.sort

summary = read_csv(summary_path)
summary_by_order = summary.to_h { |row| [row.fetch("order"), row] }
fail_audit("Expected 10 summary rows, found #{summary.length}") unless summary.length == 10

stage1 = read_csv(stage1_path)
stage1_by_order = stage1.to_h { |row| [row.fetch("order"), row] }
fail_audit("Expected 10 Stage-1 metric rows, found #{stage1.length}") unless stage1.length == 10

recomputed_orders = []
schedule.each do |row|
  pair = row.fetch("pair")
  order = row.fetch("order")
  first = row.fetch("first_target")
  second = row.fetch("second_target")
  group = order.sub("->", "_then_")
  other_classes = CLASSES - [first, second]

  first_after_first = accuracy.call(group, "stage1", first)
  first_after_baseline = accuracy.call(group, "baseline_second", first)
  first_after_retain = accuracy.call(group, "retain_previous_second", first)
  second_baseline = accuracy.call(group, "baseline_second", second)
  second_retain = accuracy.call(group, "retain_previous_second", second)
  remaining_baseline = other_classes.sum { |label| accuracy.call(group, "baseline_second", label) } / 8.0
  remaining_retain = other_classes.sum { |label| accuracy.call(group, "retain_previous_second", label) } / 8.0
  stage1_acc_s = (CLASSES - [first]).sum { |label| accuracy.call(group, "stage1", label) } / 9.0
  erase_efficacy = 1.0 - first_after_first
  stage1_h_o = (erase_efficacy + stage1_acc_s).zero? ? 0.0 : 2.0 * erase_efficacy * stage1_acc_s / (erase_efficacy + stage1_acc_s)

  expected_summary = {
    "first_target_accuracy_after_first_edit" => first_after_first,
    "first_target_accuracy_after_second_baseline" => first_after_baseline,
    "first_target_accuracy_after_retain_previous_second" => first_after_retain,
    "first_target_raw_delta_after_second_baseline" => first_after_baseline - first_after_first,
    "first_target_raw_delta_after_retain_previous_second" => first_after_retain - first_after_first,
    "second_target_final_accuracy_baseline" => second_baseline,
    "second_target_final_accuracy_retain_previous" => second_retain,
    "second_target_raw_difference_retain_minus_baseline" => second_retain - second_baseline,
    "remaining_8_mean_accuracy_baseline" => remaining_baseline,
    "remaining_8_mean_accuracy_retain_previous" => remaining_retain,
    "remaining_8_mean_raw_difference_retain_minus_baseline" => remaining_retain - remaining_baseline,
    "stage1_Acc_e" => first_after_first,
    "stage1_Acc_s" => stage1_acc_s,
    "stage1_H_o" => stage1_h_o
  }
  saved_summary = summary_by_order.fetch(order)
  expected_summary.each do |field, calculated|
    fail_audit("Summary mismatch for #{order} #{field}") unless close_enough?(calculated, saved_summary.fetch(field))
  end

  saved_stage1 = stage1_by_order.fetch(order)
  { "Acc_e" => first_after_first, "Acc_s" => stage1_acc_s, "H_o" => stage1_h_o }.each do |field, calculated|
    fail_audit("Stage-1 mismatch for #{order} #{field}") unless close_enough?(calculated, saved_stage1.fetch(field))
  end

  recomputed_orders << {
    "pair" => pair,
    "order" => order,
    "first_target" => first,
    "second_target" => second,
    "first_after_first" => first_after_first,
    "first_after_baseline" => first_after_baseline,
    "first_after_retain" => first_after_retain,
    "baseline_delta_from_stage1" => first_after_baseline - first_after_first,
    "retain_delta_from_stage1" => first_after_retain - first_after_first,
    "retain_minus_baseline_first_final" => first_after_retain - first_after_baseline,
    "second_baseline" => second_baseline,
    "second_retain" => second_retain,
    "retain_minus_baseline_second_final" => second_retain - second_baseline,
    "remaining_8_baseline" => remaining_baseline,
    "remaining_8_retain" => remaining_retain,
    "retain_minus_baseline_remaining_8" => remaining_retain - remaining_baseline
  }
end

cell_markers = Dir.glob(File.join(result_root, "raw", "cells", "**", "pair_experiment_complete.json"))
fail_audit("Expected 310 completion markers, found #{cell_markers.length}") unless cell_markers.length == 310
cell_markers.each do |marker_path|
  directory = File.dirname(marker_path)
  marker = JSON.parse(File.read(marker_path))
  predictions = File.join(directory, "predictions.csv")
  metrics = File.join(directory, "metrics.json")
  fail_audit("Missing cell predictions: #{directory}") unless File.file?(predictions)
  fail_audit("Missing cell metrics: #{directory}") unless File.file?(metrics)
  fail_audit("Prediction hash mismatch: #{directory}") unless marker.fetch("prediction_sha256") == Digest::SHA256.file(predictions).hexdigest
  fail_audit("Metrics hash mismatch: #{directory}") unless marker.fetch("metrics_sha256") == Digest::SHA256.file(metrics).hexdigest
  fail_audit("Completion marker image count mismatch: #{directory}") unless Integer(marker.fetch("n_images")) == 200
  fail_audit("Formal cleanup marker mismatch: #{directory}") unless marker.fetch("formal_images") == "deleted-after-successful-evaluation-and-qualitative-copy"
end

stage1_manifests = Dir.glob(File.join(result_root, "checkpoints", "stage1", "**", "*.manifest.json"))
stage2_manifests = Dir.glob(File.join(result_root, "checkpoints", "stage2", "**", "*.manifest.json"))
fail_audit("Expected 10 Stage-1 manifests, found #{stage1_manifests.length}") unless stage1_manifests.length == 10
fail_audit("Expected 20 Stage-2 manifests, found #{stage2_manifests.length}") unless stage2_manifests.length == 20
stage1_hashes = stage1_manifests.to_h do |path|
  data = JSON.parse(File.read(path))
  [data.fetch("first_target"), data.fetch("checkpoint_sha256")]
end
stage2_by_order = stage2_manifests.group_by { |path| File.basename(File.dirname(path)) }
stage2_by_order.each do |order_slug, paths|
  fail_audit("Expected two Stage-2 variants for #{order_slug}") unless paths.length == 2
  manifests = paths.map { |path| JSON.parse(File.read(path)) }
  first_target = manifests.first.fetch("first_target")
  expected_parent = stage1_hashes.fetch(first_target)
  parents = manifests.map { |item| item.fetch("parent_checkpoint_sha256") }.uniq
  fail_audit("Stage-2 variants do not share a parent for #{order_slug}") unless parents == [expected_parent]
end

order_effects = []
summary.group_by { |row| row.fetch("pair") }.each do |pair, pair_rows|
  pair.split("|").each do |concept|
    when_first = pair_rows.find { |row| row.fetch("first_target") == concept }
    when_second = pair_rows.find { |row| row.fetch("second_target") == concept }
    {
      "baseline" => ["first_target_accuracy_after_second_baseline", "second_target_final_accuracy_baseline"],
      "retain_previous" => ["first_target_accuracy_after_retain_previous_second", "second_target_final_accuracy_retain_previous"]
    }.each do |variant, fields|
      first_accuracy = Float(when_first.fetch(fields[0]))
      second_accuracy = Float(when_second.fetch(fields[1]))
      order_effects << {
        "pair" => pair,
        "concept" => concept,
        "variant" => variant,
        "accuracy_when_erased_first" => first_accuracy,
        "accuracy_when_erased_second" => second_accuracy,
        "raw_difference_first_minus_second" => first_accuracy - second_accuracy
      }
    end
  end
end

order_headers = %w[pair concept variant accuracy_when_erased_first accuracy_when_erased_second raw_difference_first_minus_second]
write_csv(File.join(output_dir, "order_effects.csv"), order_headers, order_effects)

order_headers = recomputed_orders.first.keys
write_csv(File.join(output_dir, "recomputed_order_summary.csv"), order_headers, recomputed_orders)

describe = lambda do |values|
  sorted = values.sort
  {
    "mean" => values.sum / values.length,
    "median" => (sorted[4] + sorted[5]) / 2.0,
    "minimum" => sorted.first,
    "maximum" => sorted.last,
    "negative_orders" => values.count(&:negative?),
    "zero_orders" => values.count(&:zero?),
    "positive_orders" => values.count(&:positive?)
  }
end

validation = JSON.parse(File.read(validation_path))
manifest = JSON.parse(File.read(manifest_path))
audit = {
  "status" => "passed",
  "source_experiment_status" => manifest.fetch("status"),
  "source_completed_at" => manifest.fetch("completed_at"),
  "protocol_fingerprint" => manifest.fetch("protocol_fingerprint"),
  "source_git_commit" => manifest.fetch("git_commit"),
  "prediction_rows" => prediction_rows,
  "evaluator_cells" => aggregates.length,
  "ordered_pairs" => schedule.length,
  "per_class_rows" => per_class_rows.length,
  "duplicate_prediction_keys" => 0,
  "maximum_probability_sum_error" => maximum_probability_sum_error,
  "per_class_recompute" => "passed",
  "summary_recompute" => "passed",
  "stage1_metrics_recompute" => "passed",
  "cell_file_hashes" => "passed",
  "stage2_shared_parent_check" => "passed",
  "formal_cleanup_markers" => "passed",
  "source_final_validation_status" => validation.fetch("status"),
  "all_predictions_sha256" => Digest::SHA256.file(predictions_path).hexdigest,
  "aggregate_descriptives" => {
    "baseline_first_target_delta_from_stage1" => describe.call(recomputed_orders.map { |row| row.fetch("baseline_delta_from_stage1") }),
    "retain_first_target_delta_from_stage1" => describe.call(recomputed_orders.map { |row| row.fetch("retain_delta_from_stage1") }),
    "retain_minus_baseline_first_target_final" => describe.call(recomputed_orders.map { |row| row.fetch("retain_minus_baseline_first_final") }),
    "retain_minus_baseline_second_target_final" => describe.call(recomputed_orders.map { |row| row.fetch("retain_minus_baseline_second_final") }),
    "retain_minus_baseline_remaining_8" => describe.call(recomputed_orders.map { |row| row.fetch("retain_minus_baseline_remaining_8") })
  }
}

File.write(File.join(output_dir, "independent_audit.json"), JSON.pretty_generate(audit) + "\n")
puts JSON.pretty_generate(audit)
