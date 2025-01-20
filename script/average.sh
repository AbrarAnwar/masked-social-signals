#!/bin/bash

# Base directory where your metrics are stored
ROOT_DIR="./metrics2/ablation_small/6_90"  # Takes the root directory as a command-line argument

# Iterate over all subdirectories in the root directory
for SUB_DIR in "$ROOT_DIR"/*; do
  if [ -d "$SUB_DIR" ]; then
    echo "Averaging subfolder: $SUB_DIR"

    # Example of running your Python script on each subfolder
    python -m evaluation.evaluate  --metric_dir="$SUB_DIR" --job=average

    # Check if the command was successful
    if [ $? -ne 0 ]; then
      echo "An error occurred while processing $SUB_DIR"
    fi
  fi
done