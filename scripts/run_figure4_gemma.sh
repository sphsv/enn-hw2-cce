#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
steps="${MAX_STEPS:-700}"
seeds="${SEEDS:-7 19 42 101 20241113}"

for implementation in cce torch_compile; do
  for seed in $seeds; do
    stem="training_figure4_gemma_${implementation}_seed${seed}"
    raw_log="$project_root/results/raw/${stem}.csv"
    output_dir="$project_root/checkpoints/figure4_gemma/${implementation}_seed${seed}"
    mkdir -p "$output_dir"

    if [[ -s "$raw_log" ]] && awk -F, -v expected="$steps" 'NR > 1 && $2 != "" {count++} END {exit count == expected ? 0 : 1}' "$raw_log"; then
      echo "Skipping completed $implementation seed=$seed"
      continue
    fi

    RAW_LOG="$raw_log" "$project_root/scripts/run_training.sh" "$implementation" \
      --model_name gemma2 \
      --dataset_name alpaca \
      --attn_impl eager \
      --output_dir "$output_dir" \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --max_steps "$steps" \
      --learning_rate 2e-5 \
      --seed "$seed" \
      --data_seed "$seed" \
      --logging_steps 1 \
      --save_strategy no \
      --eval_strategy no \
      --report_to none \
      --bf16 true \
      --gradient_checkpointing true \
      --cross_entropy_impl "$implementation"
  done
done
