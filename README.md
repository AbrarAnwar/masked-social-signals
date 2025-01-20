## Installation
Git clone the the ```bite``` branch. Then, create your conda environment by

```conda create -n [name] python=3.9```

Install dependencies by

```pip install -r requirements.txt```

## Dataset and Pretrained VQVAE

- [Download preprocessed dining dataset]()
- [Download pretrained VQVAE weights for gaze, headpose, and pose]()

## Experiment
Experiment logging through ```wandb``` and grid search through ```wandb.sweep```

### M3PT transformer

Check the ```script/train.sh``` file and run under the root directory using ```sbatch```

```sbatch script/train.sh```

Modify parameters in ```cfgs/masktransformer.yaml``` as needed

### VQVAE pre-training

Check the ```script/train_vqvae.sh``` file and run under the root directory using ```sbatch```

```sbatch script/train_vqvae.sh```

Modify parameters in ```cfgs/vqvae.yaml``` as needed

Save the weights by ```bash script/save.sh``` and make sure the module path is correct

# Ablation

Check the ```script/ablation.sh``` file and run under the root directory using ```sbatch```

```sbatch script/train.sh```

Modify ```segment``` and ```segment_length``` in ```cfgs/masktransformer.yaml``` to fit the ablation citeria

# Evaluation
Make sure the path is correct in the following bash files

Main result: ```bash script/evaluate_kfold.sh```
Ablation result: ```bash script/evaluate_ablation.sh```
Averaging over kfolds: ```bash script/average.sh```

