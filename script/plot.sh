MODULE_PAHT="./checkpoints/transformer/main/pose_only/masktransformertest_idx=30/epoch=148-val_loss=0.0028.ckpt"
RESULT_DIR="./plots/pose_only/30"
TEST_IDX=30

python -m evaluation.evaluate --job=plot --module_path="$MODULE_PAHT" --result_dir="$RESULT_DIR" --test_idx="$TEST_IDX"