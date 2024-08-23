from evaluation.evalutils import make_video, save_model, evaluate, average_metrics
from utils.dataset import get_loaders
import argparse, os, json
from pytorch_lightning import seed_everything



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--module_path", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--pretrained_dir", type=str)
    parser.add_argument("--metric_dir", type=str, help='Directory to save metrics')
    parser.add_argument("--metric_result", type=str, help='Directory to averaged metrics for averaging')
    parser.add_argument("--job", type=str)
    parser.add_argument("--transformer", '-tf', action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_idx", type=int, default=30)

    return parser.parse_args()

def save_metrics(metrics, args):
    if 'seed' in args.module_path:
        metric_dir = f"{args.metric_dir}/seed/{metrics['feature_mask']}"
        os.makedirs(metric_dir, exist_ok=True)
        metric_path = f'{metric_dir}/{args.seed}.json'

    elif 'test_idx' in args.module_path:
        metric_dir = f"{args.metric_dir}/kfold/{metrics['feature_mask']}"
        os.makedirs(metric_dir, exist_ok=True)
        metric_path = f'{metric_dir}/{args.test_idx}.json'

    with open(metric_path, 'w') as f:
        json.dump(metrics, f)

    print(f'Metrics saved at {metric_path}')


def main():
    args = get_args()
    if args.job == 'average':
        average_metrics(args.metric_result)
        return
    seed_everything(args.seed)
    print(f'Test idx: {args.test_idx}')
    _, _, test_loader = get_loaders(batch_path='/home/tangyimi/masked-social-signals/dining_dataset/batch_window36_stride18_v4', 
                                        test_idx=args.test_idx, 
                                        batch_size=32, 
                                        num_workers=2)

    if args.job == 'save':
        save_model(args.module_path, args.pretrained_dir)
    elif args.job == 'evaluate':
        metrics = evaluate(args.module_path, test_loader)
        if args.metric_dir:
            save_metrics(metrics, args)
    
    elif args.job == 'visualize':
        make_video(module_path=args.module_path, 
                result_dir=args.result_dir, 
                dataloader=test_loader, 
                transformer=args.transformer)
    else:
        raise NotImplementedError('Job not supported')

if __name__ == '__main__':
    main()
    