from experiment.module import VQVAE_Module, AutoEncoder_Module
from utils.dataset import get_loaders
from utils.visualize import visualize
from evaluation.metric import PCK, FID, W1
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
import argparse, wandb, torch, os
from tqdm import tqdm
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--pretrained_dir", type=str)

    return parser.parse_args()

def pretty_print(PCK, FID, W1):
    print("+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'Task':<10} | {'FID':<10} | {'W1_vel':<10} | {'W1_acc':<10} | {'PCK':<10} |")
    print("+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+")

    # Iterate through the metrics
    for task in ['headpose', 'gaze', 'pose']:
        # Fetch each metric with limited decimal places, use 'N/A' if not present
        fid = f"{FID[task].get_averages()['FID']:.4f}" if task in FID else 'N/A    '
        w1_vel = f"{W1[task].get_averages()['W1_vel']:.4f}" if task in FID else 'N/A    '
        w1_acc = f"{W1[task].get_averages()['W1_acc']:.4f}" if task in FID else 'N/A    '
        pck = f"{PCK.get_averages()['pck']:.4f}" if task =='pose' else 'N/A    '

        print(f"| {task:<10} | {fid:<10} | {w1_vel:<10} | {w1_acc:<10} | {pck:<10} |")
        print("+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+")


def evaluate(model_path):
    tasks = ['headpose','gaze','pose', 'speaker','bite']

    #model = MaskTransformer.load_from_checkpoint(model_path)
    model = VQVAE_Module.load_from_checkpoint(model_path)
    pck = PCK()
    fids = {task:FID() for task in tasks}
    w1s = {task:W1() for task in tasks}

    speaker_total = 0
    bite_total = 0
    speaker_1s = 0
    bite_1s = 0

    # testing on the best model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            ys, preds, _ = model.forward(batch)

            for task_idx, task in enumerate(tasks[:-2]):
                y = ys[task_idx]
                pred = preds[task_idx]
                
                # undo normalization
                y_undo = model.normalizer.minmax_denormalize(y, task)
                pred_undo = model.normalizer.minmax_denormalize(pred, task)

                if task == 'pose':
                    pck(y_undo, pred_undo)
                    fids[task](y, pred)
                else:
                    fids[task](y_undo, pred_undo)

                w1s[task](y_undo, pred_undo)
    pretty_print(pck, fids, w1s)



def test(model_path, result_dir):
    model = VQVAE_Module.load_from_checkpoint(model_path)

    os.makedirs(result_dir, exist_ok=True)

    test_losses = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader)):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            y, y_hat, _ = model.forward(batch)
            loss = F.mse_loss(y_hat, y, reduction='mean')
            test_losses.append(loss)
            if batch_idx >=  len(val_dataloader) - 2:
                file_name = f'{result_dir}/{batch_idx}'
                visualize(model.task, model.normalizer, y, y_hat, file_name)

    print(f"Test loss: {torch.stack(test_losses).mean()}")

def save_model(model_path, pretrain_path):
    os.makedirs(pretrain_path, exist_ok=True)
    if 'autoencoder' in model_path:
        model = AutoEncoder_Module.load_from_checkpoint(model_path)
        model.autoencoder.save(f"{pretrain_path}/autoencoder.pth")
    elif 'vqvae' in model_path:
        model = VQVAE_Module.load_from_checkpoint(model_path)
        model.model.save(f"{pretrain_path}/vqvae.pth")

    

if __name__ == '__main__':
    _, val_dataloader, test_dataloader = get_loaders(batch_path='/home/tangyimi/masked-social-signals/dining_dataset/batch_window36_stride18_v4', 
                                                validation_idx=30, 
                                                batch_size=32, 
                                                num_workers=2)
    args = get_args()
    #test(args.model_path, args.result_dir)
    save_model(args.model_path, args.pretrained_dir)
    #evaluate(args.model_path)
    