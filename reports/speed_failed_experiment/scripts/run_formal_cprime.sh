#!/bin/bash
set -euo pipefail

TARGET="Snoopy"
ANCHOR=" "
TARGET_DIR="Snoopy"
CONTENTS="Snoopy, Mickey, Spongebob, Pikachu, Hello Kitty"
ROOT="${ROOT:-logs/formal_cprime}"
GPU_ID="${GPU_ID:-}"
SD_CKPT="${SD_CKPT:-CompVis/stable-diffusion-v1-4}"
RETAIN_SCALE="${RETAIN_SCALE:-1.0}"
LAMB="${LAMB:-0.5}"
AUG_NUM="${AUG_NUM:-10}"
THRESHOLD="${THRESHOLD:-1e-1}"
CLEAN_IMAGES_AFTER_METRICS="${CLEAN_IMAGES_AFTER_METRICS:-0}"

need_pretrain_instance() {
  for concept in Snoopy Mickey Spongebob Pikachu "Hello Kitty"; do
    local dir="data/pretrain/instance/$concept/original"
    if [ ! -d "$dir" ] || [ "$(find "$dir" -maxdepth 1 -name '*.png' | wc -l)" -lt 800 ]; then
      return 0
    fi
  done
  return 1
}

need_pretrain_coco() {
  local dir="data/pretrain/coco/coco/original"
  [ ! -d "$dir" ] || [ "$(find "$dir" -maxdepth 1 -name '*.png' | wc -l)" -lt 1000 ]
}

record_has_metrics() {
  local record="$1"
  shift
  [ -f "$record" ] || return 1
  local content
  for content in "$@"; do
    grep -q "^$content: CS is " "$record" || return 1
  done
  return 0
}

run_gpu() {
  if [ -n "$GPU_ID" ]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$@"
  else
    "$@"
  fi
}

prepare_pretrain() {
  if need_pretrain_instance; then
    run_gpu python sample.py \
      --sd_ckpt "$SD_CKPT" \
      --erase_type "instance" \
      --target_concept "instance" \
      --contents "$CONTENTS" \
      --mode "original" \
      --num_samples 10 --batch_size 10 \
      --save_root "data/pretrain"
  fi

  if need_pretrain_coco; then
    run_gpu python sample2.py \
      --sd_ckpt "$SD_CKPT" \
      --erase_type "coco" \
      --target_concept "coco" \
      --contents "coco" \
      --mode "original" \
      --num_samples 1 --batch_size 10 \
      --save_root "data/pretrain"
  fi

  if ! record_has_metrics "data/pretrain/instance/record_metrics.txt" Snoopy Mickey Spongebob Pikachu "Hello Kitty"; then
    run_gpu python src/clip_score_cal.py \
      --contents "$CONTENTS" \
      --root_path "data/pretrain/instance" \
      --sub_root "original" \
      --pretrained_path "data/pretrain/instance"
  fi

  if ! record_has_metrics "data/pretrain/coco/record_metrics.txt" coco; then
    run_gpu python src/clip_score_cal.py \
      --contents "coco" \
      --root_path "data/pretrain/coco" \
      --sub_root "original" \
      --pretrained_path "data/pretrain/coco"
  fi
}

cleanup_mode_images() {
  local mode_root="$1"
  if [ "$CLEAN_IMAGES_AFTER_METRICS" != "1" ]; then
    return
  fi
  for content in Snoopy Mickey Spongebob Pikachu "Hello Kitty" coco; do
    rm -rf "$mode_root/$TARGET_DIR/$content/edit" "$mode_root/$TARGET_DIR/$content/combine"
  done
}

run_mode() {
  local mode_name="$1"
  local baseline="$2"
  local mode_root="$ROOT/$mode_name"
  mkdir -p "$mode_root/$TARGET_DIR"

  if record_has_metrics "$mode_root/$TARGET_DIR/record_metrics.txt" Snoopy Mickey Spongebob Pikachu "Hello Kitty" coco; then
    return
  fi

  if [ ! -f "$mode_root/$TARGET_DIR/weight.pt" ]; then
    run_gpu python train_erase_null.py \
      --sd_ckpt "$SD_CKPT" \
      --baseline "$baseline" \
      --target_concepts "$TARGET" --anchor_concepts "$ANCHOR" \
      --retain_path "data/instance.csv" --heads "concept" \
      --save_path "$mode_root/$TARGET_DIR" --file_name "weight" \
      --params V --aug_num "$AUG_NUM" --threshold "$THRESHOLD" \
      --retain_scale "$RETAIN_SCALE" --lamb "$LAMB" \
      --layer_map_path "$ROOT/layer_map.csv" \
      --diagnostics_path "$mode_root/$TARGET_DIR/diagnostics.json"
  fi

  run_gpu python sample.py \
    --sd_ckpt "$SD_CKPT" \
    --erase_type "instance" \
    --target_concept "$TARGET_DIR" \
    --contents "$CONTENTS" \
    --mode "edit" \
    --num_samples 10 --batch_size 10 \
    --save_root "$mode_root" \
    --edit_ckpt "$mode_root/$TARGET_DIR/weight.pt"

  run_gpu python sample2.py \
    --sd_ckpt "$SD_CKPT" \
    --erase_type "coco" \
    --target_concept "$TARGET_DIR" \
    --contents "coco" \
    --mode "edit" \
    --num_samples 1 --batch_size 10 \
    --save_root "$mode_root" \
    --edit_ckpt "$mode_root/$TARGET_DIR/weight.pt"

  run_gpu python src/clip_score_cal.py \
    --contents "$CONTENTS, coco" \
    --root_path "$mode_root/$TARGET_DIR" \
    --sub_root "edit" \
    --pretrained_path "data/pretrain/instance"

  cleanup_mode_images "$mode_root"
}

prepare_pretrain
run_mode "speed" "SPEED"
run_mode "cprime_null_no_iec" "cprime_null_no_iec"
run_mode "cprime_direct_eq_no_null" "cprime_direct_eq_no_null"

python src/aggregate_table1.py \
  --root "$ROOT" \
  --target "$TARGET_DIR" \
  --modes "original,speed,cprime_null_no_iec,cprime_direct_eq_no_null" \
  --output_csv "$ROOT/tables/cprime_table1.csv" \
  --output_json "$ROOT/tables/cprime_table1.json" \
  --layer_map "$ROOT/layer_map.csv"
