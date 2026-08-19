#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"
require "digest"
require "json"
require "pathname"
require "time"

root = Pathname.new(
  ARGV.fetch(0, File.join(__dir__, "outputs", "sequential_oce_celebrity_long_horizon_v1"))
).expand_path

def load_json(path)
  JSON.parse(path.read)
end

def csv_rows(path)
  CSV.read(path, headers: true).map(&:to_h)
end

def truthy(value)
  value == true || %w[true 1 yes].include?(value.to_s.downcase)
end

def optional_float(value)
  value.nil? || value.to_s.empty? ? nil : value.to_f
end

def accuracy(rows)
  detected = rows.count { |row| truthy(row.fetch("face_detected")) }
  correct = rows.count { |row| truthy(row.fetch("correct")) }
  {
    "sample_count" => rows.length,
    "face_detected_count" => detected,
    "no_face_count" => rows.length - detected,
    "correct_count" => correct,
    "accuracy" => detected.zero? ? nil : correct.fdiv(detected)
  }
end

def close?(left, right, tolerance = 1e-12)
  left = optional_float(left)
  right = optional_float(right)
  return left.nil? && right.nil? if left.nil? || right.nil?

  (left - right).abs <= tolerance
end

def descriptive(values)
  finite = values.compact.map(&:to_f).sort
  return { "mean" => nil, "median" => nil, "maximum" => nil } if finite.empty?

  middle = finite.length / 2
  median = if finite.length.odd?
             finite[middle]
           else
             (finite[middle - 1] + finite[middle]).fdiv(2)
           end
  {
    "mean" => finite.sum.fdiv(finite.length),
    "median" => median,
    "maximum" => finite.max
  }
end

def harmonic(acc_e, acc_s)
  return nil if acc_e.nil? || acc_s.nil?

  erase_success = 1.0 - acc_e
  return 0.0 if erase_success <= 0.0 || acc_s <= 0.0

  2.0 / ((1.0 / erase_success) + (1.0 / acc_s))
end

def digest(path)
  Digest::SHA256.file(path).hexdigest
end

manifest_path = root / "run_manifest.json"
raise "missing run manifest" unless manifest_path.file?

manifest = load_json(manifest_path)
config = manifest.fetch("config")
profile = manifest.dig("budget_profile", "selected_profile")
raise "unlocked/unknown profile" unless %w[profile_5 profile_10].include?(profile)

failures = []
checks = {}
active_fingerprint = manifest.fetch("active_protocol_fingerprint")
targets = config.fetch("targets")
retains = config.fetch("fixed_retains")
profile_seeds = config.dig("generation", "profile_seeds", profile).map(&:to_i)

checks["target_count"] = targets.length
checks["retain_count"] = retains.length
checks["target_unique"] = targets.uniq.length == 100
checks["retain_unique"] = retains.uniq.length == 100
checks["sets_disjoint"] = (targets & retains).empty?
unless checks.values_at(
  "target_count", "retain_count", "target_unique", "retain_unique", "sets_disjoint"
) == [100, 100, true, true, true]
  failures << "celebrity sets"
end

schedule_path = root / "inputs" / "target_schedule.csv"
schedule = csv_rows(schedule_path)
order_a = schedule.select { |row| row["order"] == "order_a" }
                  .sort_by { |row| row["sequence_position"].to_i }
                  .map { |row| row["concept"] }
order_b = schedule.select { |row| row["order"] == "order_b" }
                  .sort_by { |row| row["sequence_position"].to_i }
                  .map { |row| row["concept"] }
checks["schedule_rows"] = schedule.length
checks["order_a_exact"] = order_a == targets
checks["order_b_exact_reverse"] = order_b == targets.reverse
checks["orders_same_set"] = order_a.sort == order_b.sort
unless schedule.length == 200 && checks.values_at(
  "order_a_exact", "order_b_exact_reverse", "orders_same_set"
).all?
  failures << "orders"
end

retain_input = csv_rows(root / "inputs" / "retain_set.csv")
checks["retain_input_exact"] = retain_input.map { |row| row["concept"] } == retains
failures << "retain input" unless checks["retain_input_exact"]

checkpoint_manifest_paths = Dir[root.join("checkpoints", "**", "*.manifest.json")].sort.map { |path| Pathname.new(path) }
checkpoint_manifests = checkpoint_manifest_paths.map { |path| load_json(path) }
checks["checkpoint_manifest_count"] = checkpoint_manifests.length
failures << "checkpoint count" unless checkpoint_manifests.length == 41

checkpoint_manifests.each do |row|
  checkpoint = Pathname.new(row.fetch("checkpoint"))
  unless checkpoint.file? && digest(checkpoint) == row["checkpoint_sha256"] &&
         row["active_protocol_fingerprint"] == active_fingerprint &&
         row["status"] == "complete" && row.fetch("tensor_names").length == row["tensor_count"].to_i
    failures << "checkpoint integrity #{checkpoint}"
  end
end
checks["checkpoint_files_hashed"] = failures.none? { |value| value.start_with?("checkpoint integrity") }
tensor_name_sets = checkpoint_manifests.map { |row| row.fetch("tensor_names") }.uniq
checks["checkpoint_tensor_names_consistent"] = tensor_name_sets.length == 1 && !tensor_name_sets.fetch(0, []).empty?
failures << "checkpoint tensor names" unless checks["checkpoint_tensor_names_consistent"]

%w[order_a order_b].each do |order|
  sequence = order == "order_a" ? order_a : order_b
  first_hashes = {}
  %w[baseline retain_history].each do |condition|
    previous_hash = nil
    previous_path = nil
    (1..10).each do |step|
      matches = checkpoint_manifests.select do |row|
        row["order"] == order && row["condition"] == condition && row["step"].to_i == step
      end
      if matches.length != 1
        failures << "checkpoint identity #{order}/#{condition}/#{step}"
        next
      end
      row = matches.first
      expected_history = sequence.first((step - 1) * 10)
      expected_batch = sequence.slice((step - 1) * 10, 10)
      expected_retain = retains + (condition == "retain_history" ? expected_history : [])
      failures << "parent hash #{order}/#{condition}/#{step}" unless row["parent_checkpoint_sha256"] == previous_hash
      failures << "parent path #{order}/#{condition}/#{step}" unless row["parent_checkpoint"] == previous_path
      failures << "batch targets #{order}/#{condition}/#{step}" unless row["batch_targets"] == expected_batch
      failures << "history targets #{order}/#{condition}/#{step}" unless row["history_targets"] == expected_history
      failures << "retain list #{order}/#{condition}/#{step}" unless row["explicit_retain_concepts"] == expected_retain
      if condition == "baseline" && row["explicit_retain_concepts"] != retains
        failures << "baseline history leak #{order}/#{step}"
      end
      if condition == "retain_history" && row["retain_history_reference"] != "current pre-edit checkpoint W0 @ Kp"
        failures << "history reference #{order}/#{step}"
      end
      first_hashes[condition] = row["checkpoint_sha256"] if step == 1
      previous_hash = row["checkpoint_sha256"]
      previous_path = row["checkpoint"]
    end
  end
  failures << "condition W0 mismatch #{order}" unless first_hashes["baseline"] == first_hashes["retain_history"]
end
checks["checkpoint_parent_chains"] = failures.none? do |value|
  value.match?(/^(parent|checkpoint identity|batch targets)/)
end
checks["condition_same_step1"] = failures.none? { |value| value.start_with?("condition W0 mismatch") }
checks["retain_history_exact"] = failures.none? do |value|
  value.match?(/^(history targets|retain list|baseline history leak|history reference)/)
end

joint = checkpoint_manifests.select { |row| row["condition"] == "joint_100" }
checks["joint_checkpoint_count"] = joint.length
unless joint.length == 1 && joint.first["parent_checkpoint_sha256"].nil? &&
       joint.first["parent_checkpoint"].nil? && joint.first["batch_targets"] == targets &&
       joint.first["explicit_retain_concepts"] == retains
  failures << "joint reference"
end

cell_marker_paths = Dir[root.join("raw", "gcd_cells", "**", "cell_complete.json")].sort.map { |path| Pathname.new(path) }
cell_rows = []
cell_integrity = true
cell_marker_paths.each do |marker_path|
  marker = load_json(marker_path)
  cell_dir = marker_path.parent
  generation_path = cell_dir / "generation_manifest.csv"
  prediction_file = cell_dir / "predictions.csv"
  metrics_path = cell_dir / "metrics.json"
  unless generation_path.file? && prediction_file.file? && metrics_path.file?
    failures << "cell files #{cell_dir}"
    cell_integrity = false
    next
  end
  generation = csv_rows(generation_path)
  predicted = csv_rows(prediction_file)
  valid = marker["status"] == "complete" && marker["cleanup_permitted"] == true &&
          marker["active_protocol_fingerprint"] == active_fingerprint &&
          marker["formal_images_remaining"].to_i.zero? &&
          marker["row_count"].to_i == generation.length && predicted.length == generation.length &&
          generation.map { |row| row["sample_key"] } == predicted.map { |row| row["sample_key"] } &&
          digest(generation_path) == marker["generation_manifest_sha256"] &&
          digest(prediction_file) == marker["prediction_sha256"] &&
          digest(metrics_path) == marker["metrics_sha256"] &&
          Time.parse(marker["cleanup_completed_at"]) >= Time.parse(marker["predictions_validated_at"])
  unless valid
    failures << "cell integrity #{cell_dir}"
    cell_integrity = false
  end
  cell_rows.concat(predicted)
end
checks["completed_cell_markers"] = cell_marker_paths.length
checks["cell_hash_count_cleanup_integrity"] = cell_marker_paths.length == 41 && cell_integrity
failures << "cell marker count" unless cell_marker_paths.length == 41

prediction_path = root / "raw" / "all_gcd_predictions.csv"
predictions = csv_rows(prediction_path)
expected_predictions = profile == "profile_5" ? 34_800 : 45_800
checks["prediction_rows"] = predictions.length
checks["expected_prediction_rows"] = expected_predictions
keys = predictions.map { |row| row["sample_key"] }
checks["duplicate_prediction_keys"] = keys.length - keys.uniq.length
checks["aggregate_matches_cell_predictions"] = predictions == cell_rows
failures << "prediction count" unless predictions.length == expected_predictions
failures << "duplicate prediction keys" unless keys.uniq.length == keys.length
failures << "aggregate raw predictions" unless checks["aggregate_matches_cell_predictions"]

# Baseline/history use identical sample tuples, and every trajectory appearance
# of a celebrity uses the same templates and predeclared profile seeds.
%w[order_a order_b].each do |order|
  (1..10).each do |step|
    tuples = {}
    %w[baseline retain_history].each do |condition|
      tuples[condition] = predictions.select do |row|
        row["order"] == order && row["condition"] == condition && row["step"].to_i == step
      end.map do |row|
        [
          row["set"], row["celebrity"], row["template_index"].to_i,
          row["seed"].to_i, row["sample_index"].to_i, row["generator_protocol"]
        ]
      end.sort
    end
    failures << "paired samples #{order}/#{step}" unless tuples["baseline"] == tuples["retain_history"]
  end
end
targets.each do |concept|
  expected = (0..4).to_a.product(profile_seeds).sort
  observed_sets = predictions.select do |row|
    %w[order_a order_b].include?(row["order"]) && row["celebrity"] == concept &&
      row["set"] == "targets" && truthy(row["trajectory_sample"])
  end.group_by { |row| [row["order"], row["condition"], row["step"]] }
  unless observed_sets.values.all? do |rows|
    rows.map { |row| [row["template_index"].to_i, row["seed"].to_i] }.sort == expected
  end
    failures << "fixed trajectory samples #{concept}"
  end
end
checks["paired_condition_samples"] = failures.none? { |value| value.start_with?("paired samples") }
checks["fixed_trajectory_samples"] = failures.none? { |value| value.start_with?("fixed trajectory samples") }

trajectory_path = root / "trajectory_per_concept.csv"
trajectory = csv_rows(trajectory_path)
checks["trajectory_rows"] = trajectory.length
failures << "trajectory row count" unless trajectory.length == 2_200

trajectory.each do |row|
  subset = predictions.select do |prediction|
    prediction["order"] == row["order"] && prediction["condition"] == row["condition"] &&
      prediction["step"].to_i == row["step"].to_i && prediction["set"] == "targets" &&
      prediction["celebrity"] == row["concept"] && truthy(prediction["trajectory_sample"])
  end
  metric = accuracy(subset)
  sequence = row["order"] == "order_a" ? order_a : order_b
  introduction = sequence.index(row["concept"]).div(10) + 1
  checkpoint = checkpoint_manifests.find do |item|
    item["order"] == row["order"] && item["condition"] == row["condition"] && item["step"].to_i == row["step"].to_i
  end
  valid = metric["sample_count"] == row["sample_count"].to_i &&
          metric["face_detected_count"] == row["face_detected_count"].to_i &&
          metric["no_face_count"] == row["no_face_count"].to_i &&
          metric["correct_count"] == row["correct_count"].to_i &&
          close?(metric["accuracy"], row["raw_gcd_accuracy"]) &&
          row["concept_introduction_step"].to_i == introduction &&
          row["current_position_age"].to_i == row["step"].to_i - introduction &&
          checkpoint && row["checkpoint_id"] == checkpoint["checkpoint_sha256"]
  unless valid
    failures << "trajectory recompute #{row.values_at('order', 'condition', 'step', 'concept').join('/')}"
    break
  end
end

intro_lookup = trajectory.select do |row|
  row["step"].to_i == row["concept_introduction_step"].to_i
end.to_h { |row| [[row["order"], row["condition"], row["concept"]], optional_float(row["raw_gcd_accuracy"])] }
trajectory.each do |row|
  intro = intro_lookup[[row["order"], row["condition"], row["concept"]]]
  expected_success = !intro.nil? && intro <= config["introduction_success_max_accuracy"].to_f
  unless close?(intro, row["introduction_raw_gcd_accuracy"]) &&
         truthy(row["was_successfully_erased_at_introduction"]) == expected_success &&
         row["introduction_status"] == (expected_success ? "successfully_erased" : "failed_at_introduction")
    failures << "introduction flag #{row.values_at('order', 'condition', 'concept').join('/')}"
    break
  end
end
checks["trajectory_recompute"] = failures.none? { |value| value.start_with?("trajectory recompute") }
checks["introduction_flags"] = failures.none? { |value| value.start_with?("introduction flag") }

step_summary = csv_rows(root / "step_summary.csv")
checks["step_summary_rows"] = step_summary.length
failures << "step summary row count" unless step_summary.length == 40
step_summary.each do |row|
  order = row["order"]
  condition = row["condition"]
  step = row["step"].to_i
  sequence = order == "order_a" ? order_a : order_b
  current = sequence.slice((step - 1) * 10, 10)
  historical = sequence.first((step - 1) * 10)
  relevant = trajectory.select do |item|
    item["order"] == order && item["condition"] == condition && item["step"].to_i == step
  end.to_h { |item| [item["concept"], item] }
  current_values = current.map { |concept| optional_float(relevant.fetch(concept)["raw_gcd_accuracy"]) }
  history_values = historical.map { |concept| optional_float(relevant.fetch(concept)["raw_gcd_accuracy"]) }
  current_stats = descriptive(current_values)
  history_stats = descriptive(history_values)
  current_json = JSON.parse(row["current_batch_individual_raw_accuracies_json"])
  history_json = JSON.parse(row["historical_individual_raw_accuracies_json"])
  retain_rows = predictions.select do |prediction|
    prediction["order"] == order && prediction["condition"] == condition &&
      prediction["step"].to_i == step && prediction["set"] == "retains" &&
      truthy(prediction["trajectory_sample"])
  end
  retain_metric = accuracy(retain_rows)
  checkpoint = checkpoint_manifests.find do |item|
    item["order"] == order && item["condition"] == condition && item["step"].to_i == step
  end
  valid = JSON.parse(row["current_batch_targets_json"]) == current &&
          current_json.keys == current && history_json.keys == historical &&
          current.each_with_index.all? { |concept, index| close?(current_json[concept], current_values[index]) } &&
          historical.each_with_index.all? { |concept, index| close?(history_json[concept], history_values[index]) } &&
          row["number_of_historical_targets"].to_i == historical.length &&
          close?(current_stats["mean"], row["current_batch_mean_raw_accuracy"]) &&
          close?(current_stats["median"], row["current_batch_median_raw_accuracy"]) &&
          close?(current_stats["maximum"], row["current_batch_maximum_raw_accuracy"]) &&
          close?(history_stats["mean"], row["mean_historical_target_raw_accuracy"]) &&
          close?(history_stats["median"], row["median_historical_target_raw_accuracy"]) &&
          close?(history_stats["maximum"], row["maximum_historical_target_raw_accuracy"]) &&
          close?(retain_metric["accuracy"], row["retain_set_gcd_accuracy"]) &&
          retain_metric["sample_count"] == row["retain_sample_count"].to_i &&
          retain_metric["face_detected_count"] == row["retain_face_detected_count"].to_i &&
          relevant.values.sum { |item| item["sample_count"].to_i } == row["target_sample_count"].to_i &&
          checkpoint && row["checkpoint_id"] == checkpoint["checkpoint_sha256"]
  unless valid
    failures << "step summary recompute #{order}/#{condition}/#{step}"
    break
  end
end
checks["step_summary_recompute"] = failures.none? { |value| value.start_with?("step summary recompute") }

paper = csv_rows(root / "paper_checkpoint_results.csv")
checks["paper_result_rows"] = paper.length
failures << "paper row count" unless paper.length == 13
paper.each do |row|
  target_rows = predictions.select do |prediction|
    prediction["order"] == row["order"] && prediction["condition"] == row["condition"] &&
      prediction["step"].to_i == row["step"].to_i && prediction["set"] == "targets" &&
      truthy(prediction["paper_sample"])
  end
  retain_rows = predictions.select do |prediction|
    prediction["order"] == row["order"] && prediction["condition"] == row["condition"] &&
      prediction["step"].to_i == row["step"].to_i && prediction["set"] == "retains" &&
      truthy(prediction["paper_sample"])
  end
  target_metric = accuracy(target_rows)
  retain_metric = accuracy(retain_rows)
  expected_per_prompt = config.dig(
    "generation", "paper_target_samples_per_prompt", row["step"].to_i.to_s
  ).to_i
  target_streams_valid = target_rows.group_by do |prediction|
    [prediction["celebrity"], prediction["template_index"]]
  end.values.all? do |stream|
    stream.map { |prediction| prediction["sample_index"].to_i }.sort == (0...expected_per_prompt).to_a &&
      stream.all? do |prediction|
        prediction["seed"].to_i == 42 && prediction["generator_protocol"] == "official_stream_42"
      end
  end
  retain_streams_valid = retain_rows.all? do |prediction|
    prediction["seed"].to_i == 42 && prediction["sample_index"].to_i.zero? &&
      prediction["generator_protocol"] == "official_stream_42"
  end
  checkpoint = checkpoint_manifests.find do |item|
    item["order"] == row["order"] && item["condition"] == row["condition"] &&
      item["step"].to_i == row["step"].to_i
  end
  valid = target_metric["sample_count"] == 500 && retain_metric["sample_count"] == 500 &&
          target_streams_valid && retain_streams_valid &&
          target_metric["face_detected_count"] == row["target_face_detected_count"].to_i &&
          target_metric["no_face_count"] == row["target_no_face_count"].to_i &&
          retain_metric["face_detected_count"] == row["retain_face_detected_count"].to_i &&
          retain_metric["no_face_count"] == row["retain_no_face_count"].to_i &&
          close?(target_metric["accuracy"], row["official_target_gcd_accuracy"]) &&
          close?(retain_metric["accuracy"], row["official_retain_gcd_accuracy"]) &&
          close?(harmonic(target_metric["accuracy"], retain_metric["accuracy"]), row["H_o"]) &&
          checkpoint && row["checkpoint_id"] == checkpoint["checkpoint_sha256"]
  unless valid
    failures << "paper recompute #{row.values_at('order', 'condition', 'step').join('/')}"
    break
  end
end
checks["paper_recompute"] = failures.none? { |value| value.start_with?("paper recompute") }

remaining_images = Dir[root.join("images", "**", "*")].select { |path| File.file?(path) }
checks["remaining_formal_files"] = remaining_images.length
failures << "formal image cleanup" unless remaining_images.empty?

qualitative_manifest_path = root / "qualitative" / "qualitative_manifest.json"
if qualitative_manifest_path.file?
  qualitative = load_json(qualitative_manifest_path)
  raw_images = Dir[root.join("qualitative", "raw", "**", "*.png")]
  contact_sheets = Dir[root.join("qualitative", "contact_sheets", "**", "*.png")]
  checks["qualitative_status"] = qualitative["status"]
  checks["qualitative_counts"] = (
    qualitative["status"] == "complete" &&
    qualitative["sequence_positions"] == config.dig("qualitative", "sequence_positions") &&
    qualitative["seeds"] == config.dig("qualitative", "seeds") &&
    qualitative["raw_image_count"].to_i == raw_images.length &&
    qualitative["contact_sheet_count"].to_i == contact_sheets.length
  )
  listed_raw = qualitative.fetch("raw_images").to_h { |row| [row["path"], row] }
  listed_sheets = qualitative.fetch("contact_sheets").to_h { |row| [row["path"], row] }
  all_qualitative = raw_images + contact_sheets
  hashes_valid = all_qualitative.all? do |path_string|
    path = Pathname.new(path_string)
    relative = path.relative_path_from(root).to_s
    listed = listed_raw[relative] || listed_sheets[relative]
    listed && listed["bytes"].to_i == path.size && listed["sha256"] == digest(path)
  end
  checks["qualitative_hashes"] = (
    hashes_valid && listed_raw.length == raw_images.length &&
    listed_sheets.length == contact_sheets.length
  )
  failures << "qualitative manifest" unless checks["qualitative_counts"]
  failures << "qualitative hashes" unless checks["qualitative_hashes"]
else
  failures << "qualitative manifest missing"
end

tarball = Pathname.new(manifest.fetch("artifact_root")) / config.dig("storage", "qualitative_tarball_name")
checks["qualitative_tarball_exists"] = tarball.file?
failures << "qualitative tarball" unless tarball.file?

payload = {
  "status" => failures.empty? ? "passed" : "failed",
  "audited_at" => Time.now.utc.iso8601,
  "active_protocol_fingerprint" => active_fingerprint,
  "budget_profile" => profile,
  "checks" => checks,
  "failures" => failures.uniq,
  "all_predictions_sha256" => digest(prediction_path),
  "qualitative_tarball_sha256" => tarball.file? ? digest(tarball) : nil
}
(root / "independent_audit.json").write(JSON.pretty_generate(payload) + "\n")

unless failures.empty?
  warn "Audit failed: #{failures.uniq.first(10).join('; ')}"
  exit 1
end

puts "Independent audit passed: #{predictions.length} predictions, 41 cells, 41 checkpoints"
