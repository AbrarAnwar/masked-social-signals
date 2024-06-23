for feature_mask in mask_headpose mask_gaze mask_pose mask_word mask_speaker mask_bite headpose_only gaze_only pose_only speaker_only bite_only; do
    srun --gres=gpu:a6000:8 --time 4200 python -m main --feature_mask $feature_mask
done