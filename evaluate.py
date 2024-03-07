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
    tasks = ['headpose','gaze','pose', 'speaker','bite']

    model = MaskTransformer.load_from_checkpoint(model_path)
    device = model.device
    model = model.to(device)
    pck = PCK()
    fids = {task:FID() for task in tasks}
    w1s = {task:W1() for task in tasks}

    speaker_total = 0
    bite_total = 0
    speaker_1s = 0
    bite_1s = 0

    train_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v3', 30, training=True)
    split = int(0.8 * len(train_dataset))
    train_indices = list(range(split))
    val_indices = list(range(split, len(train_dataset)))
    val_dataset = Subset(train_dataset, val_indices)

    test_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v3', 30, training=False)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    
    model.eval()
    with torch.no_grad():
        # for batch in tqdm(val_dataloader):
        #     batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        #     ys, preds = model.forward(batch)

        #     for task_idx, task in enumerate(tasks[:-2]):
        #         y = ys if single else ys[task_idx]
        #         pred = preds if single else preds[task_idx]
                
        #         # undo normalization
        #         y_undo = model.normalizer.minmax_denormalize(y, task)
        #         pred_undo = model.normalizer.minmax_denormalize(pred, task)

        #         if task == 'pose':
        #             pck(y_undo, pred_undo)
        #             fids[task](y, pred)
        #         else:
        #             fids[task](y_undo, pred_undo)

        #         w1s[task](y_undo, pred_undo)
        segment = 12
        for batch in tqdm(train_dataloader):
            speaker = batch['speaker']
            bite = batch['bite']
            bz = speaker.size(0)

            # count 1s and 0s in speaker and bite
            speaker_reshaped = speaker.reshape(bz, 3, segment, -1) # (bz, 3, 6, 180)
            speaker_sum = speaker_reshaped.sum(dim=-1) # (bz, 3, 180)
            speaker_tranformed = (speaker_sum > 0.3 * 90).float().unsqueeze(-1)

            bite_reshaped = bite.reshape(bz, 3, segment, -1) # (bz*3, 6, 180)
            bite_sum = bite_reshaped.sum(dim=-1) # (bz, 3, 180)
            bite_tranformed = (bite_sum >= 1).float().unsqueeze(-1)

            speaker_1s += speaker_tranformed.sum()
            bite_1s += bite_tranformed.sum()

            # count total
            speaker_total += speaker_tranformed.numel()
            bite_total += bite_tranformed.numel()

    pretty_print(pck, fids, w1s)
    print(f"Speaker: {speaker_1s/speaker_total:.4f}")
    print(f"Bite: {bite_1s/bite_total:.4f}")
    print(f'Speaker_1s: {speaker_1s}')
    print(f'Bite_1s: {bite_1s}')
            

if __name__ == '__main__':
    args = get_args()
    evaluate(args.model_path, single=args.single)