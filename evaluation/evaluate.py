from evaluation.evalutils import make_video, save_model, evaluate, average_metrics, plot_pose_scatter
from utils.dataset import get_loaders
import argparse, os, json
from pytorch_lightning import seed_everything



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--module_path", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--pretrained_dir", type=str)
    parser.add_argument("--metric_dir", type=str)
    parser.add_argument("--job", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_idx", type=int, default=30)
    parser.add_argument("--batch_path", type=str, default='./dining_dataset/batch_window36_stride18_v4')

    return parser.parse_args()

def save_metrics(metrics, args):
    # if 'seed' in args.module_path:
    #     metric_dir = f"{args.metric_dir}/seed/{metrics['feature_mask']}"
    #     os.makedirs(metric_dir, exist_ok=True)
    #     metric_path = f'{metric_dir}/{args.seed}.json'

    metric_dir = f"{args.metric_dir}/{metrics['feature_mask']}"
    os.makedirs(metric_dir, exist_ok=True)
    metric_path = f'{metric_dir}/{args.test_idx}.json'
    with open(metric_path, 'w') as f:
        json.dump(metrics, f)

    print(f'Metrics saved at {metric_path}')


def main():
    args = get_args()
    if args.job == 'average':
        average_metrics(args.metric_dir)
        return

    elif args.job == 'save':
        save_model(args.module_path, args.pretrained_dir)
        return

    #seed_everything(args.seed)
    print(f'Test idx: {args.test_idx}')
    _, _, test_loader = get_loaders(batch_path=args.batch_path, 
                                    test_idx=args.test_idx)

    
    if args.job == 'evaluate':
        metrics = evaluate(args.module_path, test_loader)
        if args.metric_dir:
            save_metrics(metrics, args)

    elif args.job == 'visualize':
        make_video(module_path=args.module_path, 
                result_dir=args.result_dir, 
                dataloader=test_loader)
        
    elif args.job == 'plot':
        plot_pose_scatter(module_path=args.module_path, 
                        result_dir=args.result_dir, 
                        dataloader=test_loader)

    else:
        raise NotImplementedError('Job not supported')

if __name__ == '__main__':
    main()
    