from models.lstm import *
from models.transformer import *
from lightning import seed_everything
import lightning.pytorch as pl
import argparse

def construct_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model', type=str, default=None, help='lstm, transformer')
    # hyperparameters
    parser.add_argument('--task', type=str, default=None, help='headpose, gaze, pose')
    parser.add_argument('--method', type=str, default='concat', help='concat, maxpool')
    parser.add_argument('--multi_task', '--mt', action='store_true', help='multi task')
    parser.add_argument('--mask_ratio', type=float, default=1/3, help='masking ratio (only for transformer multi task)')
    parser.add_argument('--mask_strategy', type=int, default=1, help='1=random masking, 2=masking random one at each timestep (only for transformer multi task)')
    parser.add_argument('--eval_type', type=int, default=1, help='masking setting when evaluating (only for transformer multi task)')
    parser.add_argument('--segment', type=int, default=12, help='segment length')
    parser.add_argument('--reduced_dim', type=int, default=64, help='reduced dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--pretrained', '--pt', action='store_true', help='use pretrained autoencoder')

    # optimizer
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', '--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500, help='warmup step')
    parser.add_argument('--grad_norm_clip', type=float, default=1.0, help='gradient norm clipping')
    parser.add_argument('--batch_size', '--bz', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    
    # transformer
    parser.add_argument('--n_layer', type=int, default=6, help='number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of heads')
    parser.add_argument('--activation_function', type=str, default='relu', help='activation function')
    parser.add_argument('--n_positions', type=int, default=512, help='number of position')
    parser.add_argument('--n_ctx', type=int, default=512, help='number of context')
    parser.add_argument('--attn_pdrop', type=float, default=0.1, help='attention dropout')
    parser.add_argument('--resid_pdrop', type=float, default=0.1, help='residual dropout')

    # seed
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()

    return vars(args)

def main():
    args = construct_args()
    seed_everything(args['seed'])

    torch.cuda.empty_cache()

    if args['model'] == 'lstm':
        model = LSTMModel(reduced_dim=args['reduced_dim'], 
                     hidden_size=args['hidden_size'], 
                     segment=args['segment'], 
                     task=args['task'], 
                     multi_task=args['multi_task'], 
                     method=args['method'], 
                     pretrained=args['pretrained'],
                     lr=args['lr'],
                     weight_decay=args['weight_decay'],
                     warmup_steps=args['warmup_steps'],
                     batch_size=args['batch_size'])
        
    elif args['model'] == 'transformer':
        model = MaskTransformer(hidden_size=args['hidden_size'],
                                segment=args['segment'],
                                task=args['task'],
                                multi_task=args['multi_task'],
                                mask_ratio=args['mask_ratio'],
                                mask_strategy=args['mask_strategy'],
                                eval_type=args['eval_type'],
                                pretrained=args['pretrained'],
                                lr=args['lr'],
                                weight_decay=args['weight_decay'],
                                warmup_steps=args['warmup_steps'],
                                batch_size=args['batch_size'],
                                n_layer=args['n_layer'], 
                                n_head=args['n_head'], 
                                n_inner=args['hidden_size']*4,
                                activation_function=args['activation_function'],
                                n_positions=args['n_positions'],
                                n_ctx=args['n_ctx'],
                                resid_pdrop=args['resid_pdrop'],
                                attn_pdrop=args['attn_pdrop'])
    else:
        raise NotImplementedError('model not implemented')

    trainer = pl.Trainer(max_epochs=args['epoch'], logger=True, gradient_clip_val=args['grad_norm_clip'])
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    main()
    



    
