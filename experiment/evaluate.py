from evaluation.evalutils import make_video, save_model, evaluate
from utils.dataset import get_loaders
import argparse
from pytorch_lightning import seed_everything



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--module_path", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--pretrained_dir", type=str)
    parser.add_argument("--job", type=str)
    parser.add_argument("--transformer", '-tf', action='store_true')
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)
    _, _, test_loader = get_loaders(batch_path='/home/tangyimi/masked-social-signals/dining_dataset/batch_window36_stride18_v4', 
                                        validation_idx=30, 
                                        batch_size=32, 
                                        num_workers=2)

    if args.job == 'save':
        save_model(args.module_path, args.pretrained_dir)
    elif args.job == 'evaluate':
        evaluate(args.module_path, test_loader)
    elif args.job == 'visualize':
        make_video(module_path=args.module_path, 
                result_dir=args.result_dir, 
                dataloader=test_loader, 
                transformer=args.transformer)
    else:
        raise NotImplementedError('Job not supported')

if __name__ == '__main__':
    main()
    