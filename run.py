import os
from pathlib import Path

# callbacks has a different import than older versions. using 1.9.3 for this work
from lightning.pytorch.callbacks import ModelCheckpoint 
from lightning import Trainer, seed_everything

import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import hydra

import numpy as np
import random

import utils

from lit_modules.train_module import TrainModule
from dataset.bc_dataset import BCDataset


import wandb


@hydra.main(config_path="cfgs", config_name="train")
def main(cfg):
    #################### start wandb ####################

    # disable logging if not training
    # if cfg['mode'] != 'train':
    #     cfg['train']['log'] = False

    log_data = cfg['train']['log']
    if log_data:
        run = wandb.init(
            project=cfg['wandb']['logger']['project'],
            config=cfg['train'],
            settings=wandb.Settings(show_emoji=False),
            reinit=True
        )
        wandb.run.name = cfg['wandb']['logger']['run_name']

    #################### seeds and ckpts ####################
    seed = cfg['train']['seed']
    seed_everything(seed)

    #################### create dataset and loaders ####################

    dataset = BCDataset(cfg['data']['data_path'])

    # Using local random_split due to torch version
    train_dataset, val_dataset = \
            utils.random_split(dataset, [.8, .2], 
                                          generator=torch.Generator().manual_seed(42)) 
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=32)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=32)

    #################### losses! ####################
    loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    #################### model and things ####################

    # load model based on config
    model = model.names[cfg['model']['name']](cfg['model'])

    ## optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'])

    # optimize only params that require grad, if needed
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['optimizer']['lr'])

    # lr schedulers
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'], )

    # metrics! empty for now
    # metrics = {'acc': BinaryAccuracy()}
    metrics = {}

    # lightning module
    lit_model = TrainModule(model, optimizer, loss, metrics, cfg, lr_scheduler=lr_scheduler)


    #################### create trainer + callbacks ####################

    hydra_dir = Path(os.getcwd())
    checkpoint_path = hydra_dir / 'checkpoints'

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        mode=cfg['wandb']['saver']['mode'],
        dirpath=checkpoint_path,
        filename='best',
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        gpus=1,
        accelerator='gpu',
        deterministic=True,
        callbacks=[checkpoint_callback],
        max_epochs=cfg['train']['epochs'],
        check_val_every_n_epoch=cfg['train']['validate_every_n_epoch'],
        num_sanity_val_steps=0,
    )


    #################### Train/test/validate model ####################


    if cfg['mode'] == 'train' or cfg['mode'] == 'resume':

        # is None if mode is not "resume"
        last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
        last_checkpoint = last_checkpoint_path \
            if os.path.exists(last_checkpoint_path) and cfg['mode'] == 'resume' else None

        trainer.fit(
            lit_model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        trainer.validate(
            lit_model,
            dataloaders=val_dataloader,
        )

    elif 'val' in cfg['mode']:

        if cfg['train']['ckpt_path'] != 'None':
            trainer.validate(
                lit_model,
                dataloaders=val_dataloader,
                ckpt_path=cfg['train']['ckpt_path']
            )
        else:
            print("Please specify train.ckpt_path if you need to!")
            trainer.validate(
                lit_model,
                dataloaders=val_dataloader
            )

    elif 'test' in cfg['mode']:

        if cfg['train']['ckpt_path'] != 'None':
            trainer.test(
                lit_model,
                dataloaders=test_dataloader,
                ckpt_path=cfg['train']['ckpt_path']
            )
        else:
            print("Please specify train.ckpt_path if you need to!")
            trainer.test(
                lit_model,
                dataloaders=test_dataloader,
            )


if __name__ == "__main__":
    main()