from models.lstm import *
from models.transformer import *
from models.single_transformer import *
from lightning import seed_everything
import lightning.pytorch as pl
import wandb
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

def main():
    wandb_logger = WandbLogger(project="masked-social-signals")
    hparams = wandb.config

    seed_everything(42)

    torch.cuda.empty_cache()
    
    model = SingleTransformer(hidden_size=hparams.hidden_size,
                        task=hparams.task,
                        segment=hparams.segment,
                        eval_type=hparams.eval_type,
                        pretrained=hparams.pretrained,
                        frozen=hparams.frozen,
                        lr=hparams.lr,
                        weight_decay=hparams.weight_decay,
                        warmup_ratio=hparams.warmup_ratio,
                        batch_size=hparams.batch_size,
                        alpha=hparams.alpha,
                        n_layer=hparams.n_layer,
                        n_head=hparams.n_head,
                        n_inner=hparams.hidden_size*4,
                        activation_function=hparams.activation_function,
                        n_ctx=hparams.n_ctx,
                        resid_pdrop=hparams.resid_pdrop,
                        attn_pdrop=hparams.attn_pdrop,
                        n_bundle=hparams.n_bundle)

 
    name = f'{hparams.task}_lr{hparams.lr}_wd{hparams.weight_decay}_frozen{hparams.frozen}_warmup{hparams.warmup_ratio}_alpha{hparams.alpha}'
    wandb_logger.experiment.name = name

    checkpoint_path = f'./ckpt/{hparams.task}_{hparams.model}/lr{hparams.lr}_wd{hparams.weight_decay}_frozen{hparams.frozen}_warmup{hparams.warmup_ratio}_alpha{hparams.alpha}'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=checkpoint_path,
        filename='best',
        save_top_k=1,
    )
    trainer = pl.Trainer(accelerator='gpu',
                         callbacks=[checkpoint_callback],
                         max_epochs=hparams.epoch, 
                         logger=wandb_logger,
                         num_sanity_val_steps=0,
                         strategy=DDPStrategy(find_unused_parameters=True))
    trainer.fit(model)
    
    best_model_path = checkpoint_callback.best_model_path
    best_model = SingleTransformer.load_from_checkpoint(best_model_path)
    trainer.test(best_model)


if __name__ == '__main__':
    with open('cfgs/single.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="masked-social-signals")

    wandb.agent(sweep_id, function=main)
    



    
