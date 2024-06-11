import os, torch
from lightning import seed_everything
import lightning.pytorch as pl
import wandb, yaml, argparse
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from models.lstm import LSTMModel
from models.transformer import MaskTransformer
from utils.dataset import get_loaders
from utils.normalize import Normalizer
import torch.distributed as dist

def get_search_hparams(config):
    search_hparams = []
    for k,v in config['parameters'].items():
        if 'values' in v:
            search_hparams.append(k)
    return search_hparams


def get_experiment_name(search_hparams, hparams):
    if search_hparams:
        return '_'.join([f'{k}={v}' for k,v in hparams.items() if k in search_hparams])
    return f'default_transformer'



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_mask", type=str, default='multi')
    parser.add_argument("--sweep_name", type=str)

    return parser.parse_args()



def main():
    # clear cuda cache 
    torch.cuda.empty_cache() 

    wandb.init(entity='tangyiming', project="masked-social-signals")
    wandb_logger = WandbLogger(entity='tangyiming', project="masked-social-signals")
    hparams = wandb.config
    seed_everything(hparams.seed)

    # load data
    train_loader, val_loader, test_loader = get_loaders(batch_path='/home/tangyimi/masked-social-signals/dining_dataset/batch_window36_stride18_v4', 
                                                           validation_idx=30, 
                                                           batch_size=hparams.batch_size, 
                                                           num_workers=2)
    
    # normalize data
    normalizer = Normalizer(train_loader)

    name = get_experiment_name(search_hparams, hparams)

    model = MaskTransformer(experiment_name=name,
                            hidden_size=hparams.hidden_size,
                            segment=hparams.segment,
                            frozen=hparams.frozen,
                            pretrained=hparams.pretrained,
                            feature_filling=hparams.feature_filling,
                            lr=hparams.lr,
                            weight_decay=hparams.weight_decay,
                            alpha=hparams.alpha,
                            batch_size=hparams.batch_size,
                            result_root_dir=hparams.result_root_dir,
                            normalizer=normalizer,
                            feature_mask=args.feature_mask,
                            n_layer=hparams.n_layer,
                            n_head=hparams.n_head,
                            n_inner=hparams.hidden_size*4,
                            activation_function=hparams.activation_function,
                            n_ctx=hparams.n_ctx,
                            resid_pdrop=hparams.resid_pdrop,
                            attn_pdrop=hparams.attn_pdrop,
                            n_bundle=hparams.n_bundle)

    checkpoint_path = f'./{hparams.ckpt}/{args.feature_mask}/{name}'
    experiment_name = f'{args.feature_mask}_{name}'

    wandb_logger.experiment.name = experiment_name
    
    os.makedirs(checkpoint_path, exist_ok=True)

    # save the best
    checkpoint_callback_best = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=checkpoint_path,
        filename='best',
        save_top_k=1,
    )

    # save every n epochs
    checkpoint_callback_every_n = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename='{epoch}-{val_loss:.6f}',
        every_n_epochs=10,
        save_top_k=-1,
    )
    
    # training
    trainer = pl.Trainer(callbacks=[checkpoint_callback_best, checkpoint_callback_every_n],
                        max_epochs=hparams.epoch, 
                        logger=wandb_logger,
                        num_sanity_val_steps=0,
                        strategy=DDPStrategy(find_unused_parameters=True))
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    

    wandb.finish()


if __name__ == '__main__':
    # make sure training dataset is ready
    # when testing use validation dataset
    with open('cfgs/multi.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    search_hparams = get_search_hparams(sweep_config) # for experiment name

    args = get_args()
    
    if args.sweep_name:
        sweep_config['name'] = args.sweep_name

    sweep_id = wandb.sweep(sweep=sweep_config, entity='tangyiming', project="masked-social-signals")

    wandb.agent(sweep_id, function=main, entity='tangyiming', project="masked-social-signals")
    

        
