from models.lstm import *
from models.transformer import *
from lightning import seed_everything
import lightning.pytorch as pl
import wandb
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

def main():
    wandb_logger = WandbLogger(project="masked-social-signals")
    hparams = wandb.config

    seed_everything(hparams.seed)
    torch.cuda.empty_cache()
    
    if hparams.model == 'lstm':
        model = LSTMModel(hidden_size=hparams.hidden_size,
                        segment=hparams.segment,
                        task=hparams.task,
                        multi_task=hparams.multi_task,
                        pretrained=hparams.pretrained,
                        frozen=hparams.frozen,
                        lr=hparams.lr,
                        weight_decay=hparams.weight_decay,
                        warmup_ratio=hparams.warmup_ratio,
                        alpha=hparams.alpha,
                        batch_size=hparams.batch_size)
        model_class = LSTMModel
        checkpoint_path = f'./ckpt/{hparams.model}/lr{hparams.lr}_frozen{hparams.frozen}_warmup{hparams.warmup_ratio}'
        name = f'{hparams.model}_lr{hparams.lr}_warmup{hparams.warmup_ratio}_frozen{hparams.frozen}'

    elif hparams.model == 'transformer':
        model = MaskTransformer(hidden_size=hparams.hidden_size,
                                segment=hparams.segment,
                                task=eval(hparams.task),
                                frozen=hparams.frozen,
                                mask_ratio=hparams.mask_ratio,
                                eval_type=hparams.eval_type,
                                pretrained=hparams.pretrained,
                                feature_filling=hparams.feature_filling,
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
        model_class = MaskTransformer
        checkpoint_path = f'./ckpt1216/{hparams.model}/{hparams.task}/lr{hparams.lr}_wd{hparams.weight_decay}_frozen{hparams.frozen}_warmup{hparams.warmup_ratio}_feature{hparams.feature_filling}'
        name = f'{hparams.model}1216_lr{hparams.lr}_wd{hparams.weight_decay}_frozen{hparams.frozen}_warmup{hparams.warmup_ratio}_feature{hparams.feature_filling}'
    else:
        raise NotImplementedError()
 
    wandb_logger.experiment.name = name
    
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
    best_model = model_class.load_from_checkpoint(best_model_path)
    trainer.test(best_model)

if __name__ == '__main__':
    with open('cfgs/multi.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="masked-social-signals")

    wandb.agent(sweep_id, function=main)
