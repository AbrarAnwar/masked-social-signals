from experiment.module import VQVAE_Module, AutoEncoder_Module, MaskTransformer_Module
from evaluation.visualize import visualize, plot
from evaluation.metric import PCK, W1, L2
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, MatthewsCorrCoef
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
    feature_mask = metrics['feature_mask_list']

    def format_metric(metric):
        if isinstance(metric, dict) and 'mean' in metric and 'std' in metric:
            return f"{metric['mean']:.4f} ± {metric['std']:.4f}"
        return f"{metric:.4f}"

    class_table = PrettyTable()
    class_table.field_names = ["Task", "Accuracy", "F1 Score", "Precision", "Recall", "nMCC"]

    for task_idx, task in enumerate(['speaker', 'bite']):
        accuracy = format_metric(metrics[task]['accuracy']) if feature_mask[task_idx-2] else '--'
        f1 = format_metric(metrics[task]['f1']) if feature_mask[task_idx-2] else '--'
        precision = format_metric(metrics[task]['precision']) if feature_mask[task_idx-2] else '--'
        recall = format_metric(metrics[task]['recall']) if feature_mask[task_idx-2] else '--'
        nmcc = format_metric(metrics[task]['nmcc']) if feature_mask[task_idx-2] else '--'
        class_table.add_row([task, accuracy, f1, precision, recall, nmcc])

    print(class_table)



def compute_metrics(metrics):
    result = dict()
    for task in metrics:
        result[task] = dict()
        for metric in metrics[task]:
            result[task][metric] = metrics[task][metric].compute().item()
            if metric == 'mcc':
                result[task]['nmcc'] = (result[task][metric] + 1) / 2

    return result
 

def evaluate(module_path, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = model_selection(module_path).to(device)
    tasks = module.model.task_list
    feature_mask = REVERSE_FEATURE_MASK[tuple(map(int, module.feature_mask.tolist()))]

    # classification metric
    classification_metrics = {'speaker': 
                            {'accuracy': Accuracy(task="binary").to(device), 
                            'f1': F1Score(task="binary", average='weighted').to(device),
                            'precision': Precision(task="binary").to(device),
                            'recall': Recall(task="binary").to(device),
                            'mcc': MatthewsCorrCoef(task='binary').to(device)},
                            'bite': 
                            {'accuracy': Accuracy(task="binary").to(device), 
                            'f1': F1Score(task="binary", average='weighted').to(device),
                            'precision': Precision(task="binary").to(device),
                            'recall': Recall(task="binary").to(device),
                            'mcc': MatthewsCorrCoef(task='binary').to(device)}}
    
    module.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            ys, preds = module.forward(batch)

            for task_idx, task in enumerate(tasks):
                if task in ['speaker', 'bite']:
                    if module.feature_mask[task_idx]:
                        y = ys[task_idx-4]
                        pred = preds[task_idx-4]
                        classification_metrics[task]['accuracy'].update(pred, y)
                        classification_metrics[task]['f1'].update(pred, y)
                        classification_metrics[task]['precision'].update(pred, y)
                        classification_metrics[task]['recall'].update(pred, y)
                        classification_metrics[task]['mcc'].update(pred, y)

    metrics = compute_metrics(classification_metrics)
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

        elif isinstance(dict_list[0][key], list):
            w1_vel = [d[key][0] for d in dict_list]
            w1_acc = [d[key][1] for d in dict_list]
            
            sum_dict[key] = [{'mean': np.mean(w1_vel), 'std': np.std(w1_vel)}, 
                {'mean': np.mean(w1_acc), 'std': np.std(w1_acc)}]

        else:
            values = [d[key] for d in dict_list]
            sum_dict[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

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


def save_model(module_path, pretrain_path):
    os.makedirs(pretrain_path, exist_ok=True)
    if 'autoencoder' in module_path:
        module = AutoEncoder_Module.load_from_checkpoint(module_path)
        module.model.save(f"{pretrain_path}/autoencoder.pth")
    elif 'vqvae' in module_path:
        module = VQVAE_Module.load_from_checkpoint(module_path)
        directory = f'{pretrain_path}/{module.task}'
        os.makedirs(directory, exist_ok=True)
        module.model.save(f"{directory}/vqvae.pth")
    else:
        raise NotImplementedError('module not supported')
    

def make_video(module_path, result_dir, dataloader):
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
                if isinstance(module, MaskTransformer_Module):
                    for task_idx, task in enumerate(module.model.task_list[:-3]):
                        if module.feature_mask[task_idx]:
                            current_y = y[task_idx]
                            current_y_hat = y_hat[task_idx]
                            
                            os.makedirs(f'{result_dir}/{task}', exist_ok=True)

                            file_name = f'{result_dir}/{task}/{batch_idx}'
                            visualize(task, module.normalizer, current_y, current_y_hat, file_name)

                else:
                    file_name = f'{result_dir}/{module.task}/{batch_idx}'
                    visualize(module.task, module.normalizer, y, y_hat, file_name)
 

def plot_pose_scatter(module_path, dataloader, result_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = model_selection(module_path).to(device)

    if not module.feature_mask.tolist()[2]:
        raise ValueError('Cannot plot scatter for no-pose model')

    feature_mask = REVERSE_FEATURE_MASK[tuple(map(int, module.feature_mask.tolist()))]
    
    print(f'Plotting for {feature_mask} FEATURE MASK')

    os.makedirs(result_dir, exist_ok=True)

    module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _, preds = module.forward(batch)
            
            pose_y = module.normalizer.minmax_denormalize(batch['pose'], 'pose')
            pose_pred = module.normalizer.minmax_denormalize(preds[2], 'pose')

            random_idx = np.random.randint(0, 3)

            file_name = f'{result_dir}/{batch_idx}_{1}'
            
            plot(pose_pred[1], pose_y[1], file_name)

    
