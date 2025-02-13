#python -m experiment.evaluate --module_path=./checkpoints/transformer/main/multi/masktransformertest_idx=30/epoch=147-val_loss=0.0015.ckpt --result_dir=results/main --test_idx=30 --job=visualize

#!/bin/bash

# Base directory where your checkpoints are stored
BASE_DIR="./checkpoints/transformer_v2/main/multi/masktransformertest_idx=30"
RESULT_DIR="./videos/transformer_v2/main"

# Iterate over all subdirectories and files
find "$BASE_DIR" -type f -name "*.ckpt" | while read -r module_path; do
  # Extract the directory containing the .ckpt file
  dir_path=$(dirname "$module_path")
  
  # Extract the folder name containing the test index (e.g., masktransformertest_idx=26)
  folder_name=$(basename "$dir_path")
  
  # Extract the test index from the folder name (e.g., 26 from masktransformertest_idx=26)
  test_idx=$(echo "$folder_name" | grep -oP 'idx=\K[0-9]+')
  
  # Check if test_idx was extracted successfully
  if [ -z "$test_idx" ]; then
    echo "Warning: Could not extract test_idx from folder name $folder_name. Skipping..."
    continue
  fi
  
  echo "Visualizing module: $module_path with test_idx=$test_idx"
  
  # Run the evaluation command with the additional --test_idx argument
  python -m evaluation.evaluate --job=visualize --module_path="$module_path" --test_idx=30 --result_dir="$RESULT_DIR/$test_idx"
  
  # Check if the command was successful
  if [ $? -ne 0 ]; then
    echo "An error occurred while visualizing $module_path with test_idx=$test_idx"
  fi
done
