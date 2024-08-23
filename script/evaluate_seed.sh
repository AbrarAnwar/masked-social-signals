# # seed 42
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/multi/best_masktransformerlr=0.0005_n_layer=12_weight_decay=1e-05/epoch=147-val_loss=0.0020.ckpt --job=evaluate --metric_dir=metrics
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/mask_gaze/default_masktransformer/epoch=136-val_loss=0.0016.ckpt --job=evaluate --metric_dir=metrics 
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/mask_headpose/default_masktransformer/epoch=147-val_loss=0.0019.ckpt --job=evaluate --metric_dir=metrics
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/mask_pose/default_masktransformer/epoch=144-val_loss=0.0016.ckpt --job=evaluate --metric_dir=metrics
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/mask_word/default_masktransformer/epoch=144-val_loss=0.0019.ckpt --job=evaluate --metric_dir=metrics
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/mask_speaker/default_masktransformer/epoch=145-val_loss=0.0024.ckpt --job=evaluate --metric_dir=metrics
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/mask_bite/default_masktransformer/epoch=146-val_loss=0.0025.ckpt --job=evaluate --metric_dir=metrics
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/gaze_only/default_masktransformer/epoch=138-val_loss=0.0047.ckpt --job=evaluate --metric_dir=metrics
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/headpose_only/default_masktransformer/epoch=144-val_loss=0.0023.ckpt --job=evaluate --metric_dir=metrics
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/pose_only/default_masktransformer/epoch=146-val_loss=0.0038.ckpt --job=evaluate --metric_dir=metrics
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/speaker_only/default_masktransformer/epoch=149-val_loss=0.0000.ckpt --job=evaluate --metric_dir=metrics
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed42/bite_only/default_masktransformer/epoch=149-val_loss=0.0000.ckpt --job=evaluate --metric_dir=metrics

# # seed 7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/multi/default_masktransformer/epoch=144-val_loss=0.0020.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/mask_gaze/default_masktransformer/epoch=144-val_loss=0.0013.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/mask_headpose/default_masktransformer/epoch=73-val_loss=0.0021.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/mask_pose/default_masktransformer/epoch=147-val_loss=0.0015.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/mask_word/default_masktransformer/epoch=145-val_loss=0.0019.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/mask_speaker/default_masktransformer/epoch=72-val_loss=0.0028.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/mask_bite/default_masktransformer/epoch=144-val_loss=0.0022.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/gaze_only/default_masktransformer/epoch=147-val_loss=0.0031.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/headpose_only/default_masktransformer/epoch=149-val_loss=0.0024.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/pose_only/default_masktransformer/epoch=148-val_loss=0.0030.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/speaker_only/default_masktransformer/epoch=107-val_loss=0.0000.ckpt --job=evaluate --metric_dir=metrics --seed=7
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed7/bite_only/default_masktransformer/epoch=149-val_loss=0.0000.ckpt --job=evaluate --metric_dir=metrics --seed=7

# seed 314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/multi/default_masktransformer/epoch=148-val_loss=0.0018.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/mask_gaze/default_masktransformer/epoch=143-val_loss=0.0012.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/mask_headpose/default_masktransformer/epoch=70-val_loss=0.0021.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/mask_pose/default_masktransformer/epoch=145-val_loss=0.0014.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/mask_word/default_masktransformer/epoch=74-val_loss=0.0023.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/mask_speaker/default_masktransformer/epoch=146-val_loss=0.0024.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/mask_bite/default_masktransformer/epoch=144-val_loss=0.0027.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/gaze_only/default_masktransformer/epoch=148-val_loss=0.0042.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/headpose_only/default_masktransformer/epoch=146-val_loss=0.0019.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/pose_only/default_masktransformer/epoch=137-val_loss=0.0033.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/speaker_only/default_masktransformer/epoch=149-val_loss=0.0000.ckpt --job=evaluate --metric_dir=metrics --seed=314
python -m experiment.evaluate --module_path=./checkpoints/transformer_vq/seed314/bite_only/default_masktransformer/epoch=149-val_loss=0.0000-v1.ckpt --job=evaluate --metric_dir=metrics --seed=314
