#!/bin/bash

# Base directory where your metrics are stored
ROOT_DIR="./metrics/kfold"  # Takes the root directory as a command-line argument


# Iterate over all subdirectories in the root directory
for SUB_DIR in "$ROOT_DIR"/*; do
  if [ -d "$SUB_DIR" ]; then
    echo "Processing subfolder: $SUB_DIR"

    # Example of running your Python script on each subfolder
    python -m experiment.evaluate  --metric_result="$SUB_DIR" --job=average

    # Alternatively, if you want to run the evaluate command directly:
    # python -m experiment.evaluate --module_path="$SUB_DIR" --job=evaluate --metric_dir=metrics

    # Check if the command was successful
    if [ $? -ne 0 ]; then
      echo "An error occurred while processing $SUB_DIR"
    fi
  fi
done
