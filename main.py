from models.lstm import *
from models.transformer import *
import argparse
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def construct_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model', type=str, default=None, help='lstm, transformer')
    # hyperparameters
    parser.add_argument('--task', type=str, default=None, help='word, headpose, gaze, pose')
    parser.add_argument('--method', type=str, default=None, help='concat, maxpool')
    parser.add_argument('--multi_task', '--mt', action='store_true', help='multi task')
    parser.add_argument('--mask_ratio', type=float, default=1/3, help='masking ratio (only for transformer multi task)')
    parser.add_argument('--mask_strategy', type=int, default=1, help='1=random masking, 2=masking random one at each timestep (only for transformer multi task)')
    parser.add_argument('--segment', type=int, default=12, help='segment length')
    parser.add_argument('--reduced_dim', type=int, default=64, help='reduced dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', '--wd', type=float, default=1e-5, help='weight decay')
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
    set_seed(args['seed'])

    # load data
    single_task = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18') 
    
    train_size = int(0.8 * len(single_task))
    train_dataset = Subset(single_task, range(0, train_size))
    test_dataset = Subset(single_task, range(train_size, len(single_task)))

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

    torch.cuda.empty_cache()
    # load model
    if args['model'] == 'lstm':
        model = LSTMModel(reduced_dim=args['reduced_dim'], 
                     hidden_size=args['hidden_size'], 
                     segment=args['segment'], 
                     task=args['task'], 
                     multi_task=args['multi_task'], 
                     method=args['method'], 
                     lr=args['lr'],
                     weight_decay=args['weight_decay'])
        
    elif args['model'] == 'transformer':
        model = MaskTransformer(hidden_size=args['hidden_size'],
                                segment=args['segment'],
                                task=args['task'],
                                multi_task=args['multi_task'],
                                mask_ratio=args['mask_ratio'],
                                mask_strategy=args['mask_strategy'],
                                lr=args['lr'],
                                weight_decay=args['weight_decay'],
                                n_layer=args['n_layer'], # 3 # 6
                                n_head=args['n_head'], # 1 # 8
                                n_inner=args['hidden_size']*4,
                                activation_function=args['activation_function'],
                                n_positions=args['n_positions'],
                                n_ctx=args['n_ctx'],
                                resid_pdrop=args['resid_pdrop'],
                                attn_pdrop=args['attn_pdrop'])
    else:
        raise NotImplementedError('model not implemented')

    trainer = pl.Trainer(max_epochs=args['epoch'], strategy=DDPStrategy(find_unused_parameters=True), logger=True)
    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == '__main__':
    main()
    



    
