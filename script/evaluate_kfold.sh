#!/bin/bash

BASE_DIR="checkpoints/v10/multi"
MERTIC_DIR="./metrics/test"

# Iterate over all subdirectories and files
find "$BASE_DIR" -type f -name "*.ckpt" | while read -r module_path; do
  # Extract the directory containing the .ckpt file
  dir_path=$(dirname "$module_path")
  
  # Extract the folder name containing the test index (e.g., masktransformertest_idx=26)
  folder_name=$(basename "$dir_path")
  
  # Extract the test index from the folder name (e.g., 26 from masktransformertest_idx=26)
  # test_idx=$(echo "$folder_name" | grep -oP 'idx=\K[0-9]+')
  
  # # Check if test_idx was extracted successfully
  # if [ -z "$test_idx" ]; then
  #   echo "Warning: Could not extract test_idx from folder name $folder_name. Skipping..."
  #   continue
  # fi
  
  echo "Evaluating module: $module_path with test_idx=30"
  
  python -m evaluation.evaluate --module_path="$module_path" --test_idx=30 --job=evaluate # --metric_dir="$MERTIC_DIR"
  
  # Check if the command was successful
  if [ $? -ne 0 ]; then
    echo "An error occurred while evaluating $module_path with test_idx=$test_idx"
  fi
done
