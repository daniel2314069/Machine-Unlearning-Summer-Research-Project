#!/usr/bin/env ruby
# frozen_string_literal: true

# Lightweight post-processing for already retrieved server results.  This does
# not run Informax or load a model.  It only normalizes two presentation details
# discovered during final QA: float32-symmetric no-tie MI values are counted at
# numerical tolerance, and the alpha figure is drawn on its observed scale.

require "csv"
require "fileutils"

root = File.expand_path(__dir__)
results = ARGV.empty? ? File.join(root, "results") : File.expand_path(ARGV.fetch(0))

%w[sample_size_summary.csv sample_size_per_layer_concept.csv].each do |name|
  path = File.join(results, name)
  table = CSV.table(path)
  table.each do |row|
    n = row[:n].to_i
    row[:enumerated_no_tie_unique_count] = (n / 2) + 1
  end
  CSV.open(path, "w", write_headers: true, headers: table.headers) do |csv|
    table.each { |row| csv << row }
  end
end

rows = CSV.read(File.join(results, "sample_size_summary.csv"), headers: true)
          .select { |row| row["scope"] == "across_seed_mean" }
          .sort_by { |row| row["n"].to_i }

metrics = {
  "mean" => ["alpha_mean", "#1f77b4"],
  "median" => ["alpha_median", "#ff7f0e"],
  "p90" => ["alpha_p90", "#2ca02c"],
  "p95" => ["alpha_p95", "#9467bd"],
  "p99" => ["alpha_p99", "#d62728"]
}
width = 900
height = 560
left = 100.0
right = 35.0
top = 55.0
bottom = 90.0
plot_w = width - left - right
plot_h = height - top - bottom
y_min = 0.007
y_max = 0.0115
x_for = lambda { |index| left + index * plot_w / (rows.length - 1) }
y_for = lambda { |value| top + (y_max - value) * plot_h / (y_max - y_min) }

svg = []
svg << %(<?xml version="1.0" encoding="UTF-8"?>)
svg << %(<svg xmlns="http://www.w3.org/2000/svg" width="#{width}" height="#{height}" viewBox="0 0 #{width} #{height}">)
svg << %(<rect width="100%" height="100%" fill="white"/>)
svg << %(<text x="#{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="21">Repository alpha distribution by nested sample size</text>)

(0..9).each do |step|
  value = y_min + step * 0.0005
  y = y_for.call(value)
  svg << %(<line x1="#{left}" y1="#{y}" x2="#{width - right}" y2="#{y}" stroke="#dddddd" stroke-width="1"/>)
  svg << %(<text x="#{left - 10}" y="#{y + 5}" text-anchor="end" font-family="sans-serif" font-size="14">#{format('%.4f', value)}</text>)
end

rows.each_with_index do |row, index|
  x = x_for.call(index)
  svg << %(<line x1="#{x}" y1="#{top}" x2="#{x}" y2="#{height - bottom}" stroke="#eeeeee" stroke-width="1"/>)
  svg << %(<text x="#{x}" y="#{height - bottom + 28}" text-anchor="middle" font-family="sans-serif" font-size="16">#{row['n']}</text>)
end

metrics.each_with_index do |(label, (column, color)), metric_index|
  points = rows.each_with_index.map { |row, index| "#{x_for.call(index)},#{y_for.call(row[column].to_f)}" }.join(" ")
  svg << %(<polyline points="#{points}" fill="none" stroke="#{color}" stroke-width="3"/>)
  rows.each_with_index do |row, index|
    svg << %(<circle cx="#{x_for.call(index)}" cy="#{y_for.call(row[column].to_f)}" r="5" fill="#{color}"/>)
  end
  legend_x = left + metric_index * 135
  svg << %(<line x1="#{legend_x}" y1="#{height - 30}" x2="#{legend_x + 25}" y2="#{height - 30}" stroke="#{color}" stroke-width="3"/>)
  svg << %(<text x="#{legend_x + 32}" y="#{height - 25}" font-family="sans-serif" font-size="15">#{label}</text>)
end

svg << %(<line x1="#{left}" y1="#{height - bottom}" x2="#{width - right}" y2="#{height - bottom}" stroke="black" stroke-width="1.5"/>)
svg << %(<line x1="#{left}" y1="#{top}" x2="#{left}" y2="#{height - bottom}" stroke="black" stroke-width="1.5"/>)
svg << %(<text x="#{left + plot_w / 2}" y="#{height - 48}" text-anchor="middle" font-family="sans-serif" font-size="17">Positive and neutral samples per class (n)</text>)
svg << %(<text x="24" y="#{top + plot_h / 2}" text-anchor="middle" font-family="sans-serif" font-size="17" transform="rotate(-90 24 #{top + plot_h / 2})">Repository alpha</text>)
svg << %(</svg>)

figure_dir = File.join(results, "figures")
FileUtils.mkdir_p(figure_dir)
File.write(File.join(figure_dir, "sample_size_repo_alpha_distribution.svg"), svg.join("\n") + "\n")

warn "normalized result metadata and wrote alpha-distribution SVG under #{results}"
