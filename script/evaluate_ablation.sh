#!/bin/bash

BASE_DIR="./checkpoints/v22"
MERTIC_DIR="./metrics/ablation2/6_90"
BATCH_PATH="./dining_dataset/batch_window6_stride3_v4"

# Iterate over all subdirectories and files
find "$BASE_DIR" -type f -name "*.ckpt" | while read -r module_path; do
  # Extract the directory containing the .ckpt file
  dir_path=$(dirname "$module_path")
  
  # Extract the test index from the folder name (e.g., 26 from masktransformertest_idx=26)
  # test_idx=$(basename "$dir_path" | grep -oP 'idx=\K[0-9]+')
  
  # # Check if test_idx was extracted successfully
  # if [ -z "$test_idx" ]; then
  #   echo "Warning: Could not extract test_idx from folder path $dir_path. Skipping..."
  #   continue
  # fi
  
  echo "Evaluating checkpoint: $module_path with test_idx=30"
  
  python -m evaluation.evaluate --job=evaluate --module_path="$module_path" --test_idx=30 --batch_path="$BATCH_PATH" # --metric_dir="$MERTIC_DIR"
  
  # Check if the command was successful
  if [ $? -ne 0 ]; then
    echo "An error occurred while evaluating $module_path with test_idx=$test_idx"
  fi
done
