from models.lstm import *
from models.transformer import *
from models.single_transformer import *
from utils.dataset import *
from torch.utils.data import DataLoader, Subset
from evaluation.metric import *
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--single", action='store_true', help='for transformer only contains one feature')

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


def evaluate(model_path, single=False):
    tasks = ['headpose','gaze','pose']

    model = MaskTransformer.load_from_checkpoint(model_path)
    device = model.device
    model = model.to(device)
    model.on_train_start()
    model.on_test_start()
    model.eval()
    with torch.no_grad():
        pck = PCK()
        fids = {task:FID() for task in tasks}
        w1s = {task:W1() for task in tasks}

        train_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v2', 30, training=True)
        split = int(0.8 * len(train_dataset))
        train_indices = list(range(split))
        val_indices = list(range(split, len(train_dataset)))
        val_dataset = Subset(train_dataset, val_indices)

        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)

        for batch in tqdm(val_dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            ys, preds = model.forward(batch)

            for task_idx, task in enumerate(tasks):
                y = ys if single else ys[task_idx]
                pred = preds if single else preds[task_idx]
                
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
            

if __name__ == '__main__':
    #model = MaskTransformer.load_from_checkpoint('/home/tangyimi/masked_mine/ckpt/transformer/lr0.005_wd1e-05_frozenFalse_warmup0.1_featurerepeat/best.ckpt')
    #pose_model = SingleTransformer.load_from_checkpoint('/home/tangyimi/masked_mine/ckpt/pose_transformer/lr0.005_wd0.0001_frozenFalse_warmup0.2_alpha0/best.ckpt')
    #gaze_model = SingleTransformer.load_from_checkpoint('/home/tangyimi/masked_mine/ckpt/gaze_transformer/lr0.005_wd1e-05_frozenFalse_warmup0.1_alpha0/best.ckpt')
    #model = MaskTransformer.load_from_checkpoint('/home/tangyimi/masked_mine/ckpt/transformer/lr0.005_wd1e-05_frozenFalse_warmup0.1_featurerepeat/best.ckpt')
    #pose_mask_model = MaskTransformer.load_from_checkpoint('/home/tangyimi/masked_mine/ckpt/transformer/pose/lr0.005_wd1e-05_frozenFalse_warmup0.1_featurerepeat/best.ckpt')
    args = get_args()
    evaluate(args.model_path, single=args.single)