from experiment.module import VQVAE_Module, AutoEncoder_Module, MaskTransformer_Module
from utils.dataset import get_loaders
from utils.visualize import visualize
from evaluation.metric import PCK, FID, W1
from torchmetrics.classification import Accuracy, F1Score
import argparse, wandb, torch, os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from prettytable import PrettyTable


REVERSE_FEATURE_MASK = {(1, 1, 1, 1, 1, 1): 'multi',
                    (0, 1, 1, 1, 1, 1): 'mask_gaze',
                    (1, 0, 1, 1, 1, 1): 'mask_headpose',
                    (1, 1, 0, 1, 1, 1): 'mask_pose',
                    (1, 1, 1, 0, 1, 1): 'mask_word',
                    (1, 1, 1, 1, 0, 1): 'mask_speaker',
                    (1, 1, 1, 1, 1, 0): 'mask_bite',
                    (1, 0, 0, 0, 0, 0): 'gaze_only',
                    (0, 1, 0, 0, 0, 0): 'headpose_only',
                    (0, 0, 1, 0, 0, 0): 'pose_only',
                    (0, 0, 0, 0, 1, 0): 'speaker_only',
                    (0, 0, 0, 0, 0, 1): 'bite_only'}



def model_selection(module_path):
    if 'autoencoder' in module_path:
        return AutoEncoder_Module.load_from_checkpoint(module_path)
    elif 'vqvae' in module_path:
        return VQVAE_Module.load_from_checkpoint(module_path)
    elif 'transformer' in module_path:
        return MaskTransformer_Module.load_from_checkpoint(module_path)
    else:
        raise NotImplementedError('module not supported')


def print_reconstruction(PCK, FID, W1, L2):
    table = PrettyTable()
    table.field_names = ["Task", "FID", "W1_vel", "W1_acc", "L2", "PCK"]

    # Iterate through the metrics
    for task in ['gaze', 'headpose', 'pose']:
        # Fetch each metric with limited decimal places, use 'N/A' if not present
        fid = f"{FID[task].get_averages()['FID']:.4f}" if task in FID else 'N/A'
        w1_vel = f"{W1[task].get_averages()['W1_vel']:.4f}" if task in W1 else 'N/A'
        w1_acc = f"{W1[task].get_averages()['W1_acc']:.4f}" if task in W1 else 'N/A'
        l2 = f"{torch.stack(L2[task]).mean().item():.4f}" if task in L2 else 'N/A'
        pck = f"{PCK.get_averages()['pck']:.4f}" if task == 'pose' else 'N/A'

        # Add row to the table
        table.add_row([task, fid, w1_vel, w1_acc, l2, pck])

    print(table)


def print_classification(metrics):
    table = PrettyTable()
    table.field_names = ["Task", "Accuracy", "F1 Score"]

    for task in metrics:
        accuracy = metrics[task]['accuracy'].compute()
        f1 = metrics[task]['f1'].compute()
        table.add_row([task, f"{accuracy:.2f}", f"{f1:.2f}"])

    print(table)


def evaluate(module_path, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = model_selection(module_path).to(device)
    tasks = module.model.task_list
    feature_mask = REVERSE_FEATURE_MASK[tuple(map(int, module.feature_mask.tolist()))]
    
    print(f'Evaluating on {feature_mask} FEATURE MASK')
    print('NOTE that all tasks are evaluated but only selected tasks are meaningful (depending on feature mask)')
    # reconstruction metric
    pck = PCK()
    fids = {task:FID() for task in tasks}
    w1s = {task:W1() for task in tasks}
    l2 = {task:[] for task in tasks}

    # classification metric
    classification_metrics = {'speaker': 
                                {'accuracy': Accuracy(task="binary").to(device), 
                                'f1': F1Score(task="binary", average='weighted').to(device)},
                                'bite': 
                                {'accuracy': Accuracy(task="binary").to(device), 
                                'f1': F1Score(task="binary", average='weighted').to(device)}}
    
    module.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            ys, preds = module.forward(batch)

            # TODO
            for task_idx, task in enumerate(tasks):
                y = ys[task_idx]
                pred = preds[task_idx]
                if task in ['gaze', 'headpose', 'pose']:
                    # undo normalization
                    y_undo = module.normalizer.minmax_denormalize(y, task)
                    pred_undo = module.normalizer.minmax_denormalize(pred, task)

                    l2[task].append(F.mse_loss(pred, y, reduction='mean').cpu())
                    w1s[task](y_undo, pred_undo)

                    if task == 'pose':
                        pck(y_undo, pred_undo)
                        fids[task](y, pred)
                    else:
                        fids[task](y_undo, pred_undo)
                    
                
                elif task in ['speaker', 'bite']:
                    classification_metrics[task]['accuracy'](pred, y)
                    classification_metrics[task]['f1'](pred, y)

    print_reconstruction(pck, fids, w1s, l2)
    print_classification(classification_metrics)



def make_video(module_path, result_dir, dataloader, transformer=True):
    module = model_selection(module_path)

    os.makedirs(result_dir, exist_ok=True)
    
    # random visualize index
    visualize_idx = np.random.randint(0, len(dataloader))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = module.to(device)
    module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if batch_idx == visualize_idx:
                y, y_hat = module.forward(batch)[:2]
                if transformer:
                    for task_idx, task in enumerate(module.model.task_list[:-3]):
                        if module.feature_mask[task_idx]:
                            current_y = y[task_idx]
                            current_y_hat = y_hat[task_idx]
                            
                            os.makedirs(f'{result_dir}/{task}', exist_ok=True)

                            file_name = f'{result_dir}/{task}/{batch_idx}'
                            visualize(task, module.normalizer, current_y, current_y_hat, file_name)

                else:
                    file_name = f'{result_dir}/{batch_idx}'
                    visualize(module.task, module.normalizer, y, y_hat, file_name)
 

def save_model(module_path, pretrain_path):
    os.makedirs(pretrain_path, exist_ok=True)
    if 'autoencoder' in module_path:
        module = AutoEncoder_Module.load_from_checkpoint(module_path)
        module.model.save(f"{pretrain_path}/autoencoder.pth")
    elif 'vqvae' in module_path:
        module = VQVAE_Module.load_from_checkpoint(module_path)
        module.model.save(f"{pretrain_path}/vqvae.pth")
    else:
        raise NotImplementedError('module not supported')