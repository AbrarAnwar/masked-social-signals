# #!/bin/bash

# # Base directory
idx=30

base_dir="./checkpoints/vqvae_1d/ablation/4_270/${idx}"

pretrained_dir="./pretrained/ablation/4_270/${idx}"


# Iterate through all subdirectories under the base directory
for dir in "$base_dir"/*/; do
    # Find the .ckpt files within the subdirectory
    for ckpt_file in $(find "$dir" -type f -name "*.ckpt"); do
        # Run the Python script with the ckpt file as an argument
        python -m experiment.evaluate --module_path="$ckpt_file" --pretrained_dir="$pretrained_dir" --job=save
    done
done
