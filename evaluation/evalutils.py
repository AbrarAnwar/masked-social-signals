from experiment.module import VQVAE_Module, AutoEncoder_Module, MaskTransformer_Module
from evaluation.visualize import visualize, plot
from evaluation.metric import PCK, W1, L2
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
    recon_table.field_names = ["Task", "W1_vel", "W1_acc", "L2", "PCK"]
    feature_mask = metrics['feature_mask_list']

    def format_metric(metric):
        if isinstance(metric, dict) and 'mean' in metric and 'std' in metric:
            return f"{metric['mean']:.4f} ± {metric['std']:.4f}"
        return f"{metric:.4f}"

    for task_idx, task in enumerate(['gaze', 'headpose', 'pose']):
        #fid = format_metric(metrics['fid'][task]) if feature_mask[task_idx] else '--'
        w1_vel = format_metric(metrics['w1'][task][0]) if feature_mask[task_idx] else '--'
        w1_acc = format_metric(metrics['w1'][task][1]) if feature_mask[task_idx] else '--'
        l2 = format_metric(metrics['l2'][task]) if feature_mask[task_idx] else '--'
        pck = format_metric(metrics['pck']) if feature_mask[task_idx] and task == 'pose' else '--'

        recon_table.add_row([task, w1_vel, w1_acc, l2, pck])

    print(recon_table)

    class_table = PrettyTable()
    class_table.field_names = ["Task", "Accuracy", "F1 Score"]

    for task_idx, task in enumerate(['speaker', 'bite']):
        accuracy = format_metric(metrics['accuracy'][task]) if feature_mask[task_idx-2] else '--'
        f1 = format_metric(metrics['f1'][task]) if feature_mask[task_idx-2] else '--'
        class_table.add_row([task, accuracy, f1])

    print(class_table)



def compute_metrics(metrics):
    result = dict()
    result['pck'] = metrics['pck'].compute()
    #result['fid'] = {task:metrics['fids'][task].get_averages()['FID'] for task in metrics['fids']}
    result['w1'] = {task: metrics['w1s'][task].compute() for task in metrics['w1s']}
    result['l2'] = {task: metrics['l2'][task].compute() for task in metrics['l2']}

    result['accuracy'] = {task:metrics['classification'][task]['accuracy'].compute().item() for task in metrics['classification']}
    result['f1'] = {task:metrics['classification'][task]['f1'].compute().item() for task in metrics['classification']}

    return result
 

def evaluate(module_path, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = model_selection(module_path).to(device)
    tasks = module.model.task_list
    feature_mask = REVERSE_FEATURE_MASK[tuple(map(int, module.feature_mask.tolist()))]

    # reconstruction metric
    pck = PCK()
    w1s = {task:W1() for task in ['gaze', 'headpose', 'pose']}
    l2 = {task:L2() for task in ['gaze', 'headpose', 'pose']}

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
            ys, preds = module.forward(batch)[:2]

            for task_idx, task in enumerate(tasks):
                y = ys[task_idx]
                pred = preds[task_idx]
                if module.feature_mask[task_idx]:
                    if task in ['gaze', 'headpose', 'pose']:
                        # undo normalization
                        y_undo = module.normalizer.minmax_denormalize(y, task)
                        pred_undo = module.normalizer.minmax_denormalize(pred, task)

                        l2[task].update(pred, y)
                        w1s[task].update(pred_undo, y_undo)
                        #fids[task].update(pred_undo, y_undo)

                        if task == 'pose':
                            pck.update(pred_undo, y_undo)
                    
                    elif task in ['speaker', 'bite']:
                        classification_metrics[task]['accuracy'](pred, y)
                        classification_metrics[task]['f1'](pred, y)

    metrics = compute_metrics({'pck':pck, 'w1s':w1s, 'l2':l2, 'classification':classification_metrics})
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
    
    if 'autoencoder' in module_path:
        module = AutoEncoder_Module.load_from_checkpoint(module_path)
        os.makedirs(pretrain_path, exist_ok=True)
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

    
