from experiment.module import VQVAE_Module, AutoEncoder_Module, MaskTransformer_Module
from utils.visualize import visualize
from evaluation.metric import PCK, FID, W1
from torchmetrics.classification import Accuracy, F1Score
import torch, os, json
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


def print_table(metrics):
    recon_table = PrettyTable()
    recon_table.field_names = ["Task", "FID", "W1_vel", "W1_acc", "L2", "PCK"]
    feature_mask = metrics['feature_mask_list']

    # Iterate through the metrics
    for task_idx, task in enumerate(['gaze', 'headpose', 'pose']):
        fid = f"{metrics['fid'][task]:.4f}" if feature_mask[task_idx] else '--'
        w1_vel = f"{metrics['w1_vel'][task]:.4f}" if feature_mask[task_idx] else '--'
        w1_acc = f"{metrics['w1_acc'][task]:.4f}" if feature_mask[task_idx] else '--'
        l2 =  f"{metrics['l2'][task]:.4f}" if feature_mask[task_idx] else '--'
        pck = f"{metrics['pck']:.4f}" if feature_mask[task_idx] and task == 'pose' else '--'

        recon_table.add_row([task, fid, w1_vel, w1_acc, l2, pck])

    print(recon_table)

    class_table = PrettyTable()
    class_table.field_names = ["Task", "Accuracy", "F1 Score"]

    for task_idx, task in enumerate(['speaker', 'bite']):
        accuracy = f"{metrics['accuracy'][task]:.2f}" if feature_mask[task_idx-2] else '--'
        f1 = f"{metrics['f1'][task]:.2f}" if feature_mask[task_idx-2] else '--'
        class_table.add_row([task, accuracy, f1])

    print(class_table)



def compute_metrics(metrics):
    result = dict()
    result['pck'] = metrics['pck'].get_averages()['pck']
    result['fid'] = {task:metrics['fids'][task].get_averages()['FID'] for task in metrics['fids']}
    result['w1_acc'] = {task:metrics['w1s'][task].get_averages()['W1_acc'] for task in metrics['w1s']}
    result['w1_vel'] = {task:metrics['w1s'][task].get_averages()['W1_vel'] for task in metrics['w1s']}
    result['l2'] = {task:torch.stack(metrics['l2'][task]).mean().item() if metrics['l2'][task] else 0 for task in metrics['l2']}

    result['accuracy'] = {task:metrics['classification'][task]['accuracy'].compute().item() for task in metrics['classification']}
    result['f1'] = {task:metrics['classification'][task]['f1'].compute().item() for task in metrics['classification']}

    return result
 

def evaluate(module_path, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = model_selection(module_path).to(device)
    tasks = module.model.task_list
    feature_mask = REVERSE_FEATURE_MASK[tuple(map(int, module.feature_mask.tolist()))]
    
    print(f'Evaluating on {feature_mask} FEATURE MASK')

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
                if module.feature_mask[task_idx]:
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

    metrics = compute_metrics({'pck':pck, 'fids':fids, 'w1s':w1s, 'l2':l2, 'classification':classification_metrics})
    metrics['feature_mask'] = feature_mask
    metrics['feature_mask_list'] = module.feature_mask.tolist()
    print_table(metrics)
    
    return metrics

def recursive_average(dict_list):
    sum_dict = {}
    
    for key in dict_list[0]:  #
        if key in ['feature_mask', 'feature_mask_list']:
            continue  

        if isinstance(dict_list[0][key], dict):
            sum_dict[key] = recursive_average([d[key] for d in dict_list])
        else:
            values = [d[key] for d in dict_list]
            sum_dict[key] = np.mean(values)

    return sum_dict

def average_metrics(root_dir):
    all_metrics = []
    for file in os.listdir(root_dir):
        with open(f'{root_dir}/{file}', 'r') as f:
            metrics = json.load(f)
        
        all_metrics.append(metrics)

    feature_mask = all_metrics[0]['feature_mask']
    print(f'Averaging metrics for {feature_mask} FEATURE MASK')

    feature_mask_list = all_metrics[0]['feature_mask_list']

    avg_metrics = recursive_average(all_metrics)
    avg_metrics['feature_mask_list'] = feature_mask_list
    print_table(avg_metrics)



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
    
