import os, pickle, wandb, yaml, argparse
os.environ['WANDB_SILENT'] = 'true'

from lightning import seed_everything
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from experiment.module import AutoEncoder_Module, VQVAE_Module, MaskTransformer_Module
from utils.dataset import get_loaders
from utils.normalize import Normalizer
from utils.utils import get_search_hparams, get_experiment_name

ENTITY = 'tangyiming'
PROJECT = 'masked-social-signals'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--sweep", type=str)
    parser.add_argument("--feature_mask", type=str, default='multi')
    parser.add_argument("--batch_path", type=str, default='./dining_dataset/batch_window36_stride18_v4')
    parser.add_argument("--ablation", '-ab', action='store_true', help='one of ablation stuides that requires different pretrain vqvaes')

    return parser.parse_args()


def get_pretrain_dir(pretrained, test_idx, ablation, segment, segment_length):
    if ablation:
        return f'./{pretrained}/ablation/{segment}_{segment_length}/{test_idx}'
    
    return f'./{pretrained}/main/{test_idx}'


def main():
    wandb.init(entity=ENTITY, project=PROJECT)
    wandb_logger = WandbLogger(entity=ENTITY, project=PROJECT)
    hparams = wandb.config
    seed_everything(hparams.seed)
    
    # load data
    train_loader, val_loader, _ = get_loaders(batch_path=args.batch_path, 
                                                test_idx=hparams.test_idx, 
                                                batch_size=hparams.batch_size, 
                                                num_workers=2)
    
    # normalize data
    if 'test' in args.sweep:
        # testing code
        normalizer = pickle.load(open('normalizer.pkl', 'rb'))
    else:
        normalizer = Normalizer(train_loader)

    
    name = get_experiment_name(search_hparams, hparams)

    if args.model == 'autoencoder':
        module = AutoEncoder_Module(normalizer=normalizer,
                            task=hparams.task,
                            segment=hparams.segment,
                            segment_length=hparams.segment_length,
                            hidden_sizes=hparams.hidden_sizes,
                            alpha=hparams.alpha,
                            lr=hparams.lr,
                            weight_decay=hparams.weight_decay)
        sub_folder = hparams.task

        
    elif args.model == 'vqvae':
        module = VQVAE_Module(normalizer=normalizer,
                            hidden_sizes=hparams.hidden_sizes,
                            h_dim=hparams.h_dim,
                            kernel=hparams.kernel,
                            stride=hparams.stride,
                            res_h_dim=hparams.res_h_dim,
                            n_res_layers=hparams.n_res_layers,
                            n_embeddings=hparams.n_embeddings,
                            embedding_dim=hparams.embedding_dim,
                            beta=hparams.beta,
                            lr=hparams.lr,
                            weight_decay=hparams.weight_decay,
                            task=hparams.task,
                            segment=hparams.segment,
                            segment_length=hparams.segment_length)
        sub_folder = f'{hparams.test_idx}/{hparams.task}'


    elif args.model == 'masktransformer':
        print('Feature mask:', args.feature_mask)
        module = MaskTransformer_Module(normalizer=normalizer,
                            hidden_size=hparams.hidden_size,
                            segment=hparams.segment,
                            segment_length=hparams.segment_length,
                            frozen=hparams.frozen,
                            pretrained=get_pretrain_dir(hparams.pretrained, hparams.test_idx, args.ablation, hparams.segment, hparams.segment_length),
                            feature_filling=hparams.feature_filling,
                            lr=hparams.lr,
                            weight_decay=hparams.weight_decay,
                            feature_mask=args.feature_mask,
                            n_layer=hparams.n_layer,
                            n_head=hparams.n_head,
                            n_inner=hparams.hidden_size*4,
                            activation_function=hparams.activation_function,
                            n_ctx=hparams.n_ctx,
                            resid_pdrop=hparams.resid_pdrop,
                            attn_pdrop=hparams.attn_pdrop,
                            n_bundle=hparams.n_bundle)
        sub_folder = args.feature_mask
    else:
        raise NotImplementedError('model not supported')

    checkpoint_path = f'./{hparams.ckpt}/{sub_folder}/{name}'

    wandb_logger.experiment.name = name
    
    os.makedirs(checkpoint_path, exist_ok=True)

    # save the best
    checkpoint_callback_best = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=checkpoint_path,
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=1,
        save_last=True,
    )
    
    # training
    trainer = pl.Trainer(callbacks=[checkpoint_callback_best],
                        max_epochs=hparams.epoch, 
                        logger=wandb_logger,
                        num_sanity_val_steps=0,
                        strategy=hparams.strategy,
                        log_every_n_steps=1)
    
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    


if __name__ == '__main__':
    args = get_args()

    with open(f'cfgs/{args.model}.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    search_hparams = get_search_hparams(sweep_config) # for experiment name
    
    if args.sweep:
        sweep_config['name'] = args.sweep
    sweep_id = wandb.sweep(sweep=sweep_config, entity=ENTITY, project=PROJECT)
    wandb.agent(sweep_id, function=main, entity=ENTITY, project=PROJECT)
    

        
