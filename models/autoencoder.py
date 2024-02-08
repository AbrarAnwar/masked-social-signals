import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy

from utils.dataset import *
from torch.utils.data import DataLoader, Subset
from utils.normalize import *
from utils.visualize import *
from utils.utils import *
import wandb
from lightning import seed_everything
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import yaml

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation=True):
        super(Encoder, self).__init__()

        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        if not activation:
            layers.pop()

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_sizes):
        super(Decoder, self).__init__()

        layers = []
        prev_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class AutoEncoder(LightningModule):
    FEATURES = {'headpose':2, 'gaze':2, 'pose':26}

    def __init__(self, task, segment, hidden_sizes, alpha, lr, weight_decay, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.batch_length = 1080
        self.segment = segment
        self.segment_length = int(self.batch_length / (2 * self.segment)) if self.task == 'pose' else int(self.batch_length / self.segment)
        self.hidden_size = copy.deepcopy(hidden_sizes)
        self.feature_dim = self.FEATURES[self.task]
        self.input_dim = self.segment_length * self.feature_dim
        self.encoder = Encoder(self.FEATURES[self.task] * self.segment_length, self.hidden_size)
        self.decoder = Decoder(self.segment_length * self.FEATURES[self.task], self.hidden_size[::-1])

        self.alpha = alpha
    
        self.lr = lr
        self.weight_decay = weight_decay
        #self.warmup_ratio = warmup_ratio

        self.batch_size = batch_size
        #dataset = MultiDataset('/data/tangyimi/batch_window36_stride18') 
        train_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v2', 30, training=True)

        split = int(0.8 * len(train_dataset))
        train_indices = list(range(split))
        val_indices = list(range(split, len(train_dataset)))

        self.train_dataset = Subset(train_dataset, train_indices)
        self.val_dataset = Subset(train_dataset, val_indices)
        self.test_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v2', 30, training=False)
        
        self.normalizer = Normalizer(self.train_dataloader())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)  
    
    '''
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_lambda = lambda epoch: self.lr_schedule(self.global_step)
        scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [scheduler]

    def lr_schedule(self, current_step):
        total_steps = self.trainer.num_training_batches * self.trainer.max_epochs
        warmup_steps = self.warmup_ratio * total_steps
        if current_step < warmup_steps:
            lr_mult = float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return self.lr * lr_mult
    '''
    
    def on_train_start(self):
        self.train_losses = []

    def forward(self, batch):
        batch[self.task] = self.normalizer.minmax_normalize(batch[self.task], self.task)

        if self.task == 'pose':
            batch[self.task] = smoothing(batch[self.task], self.batch_length).to(self.device)

        batch = batch[self.task]
        bz = batch.size(0)

        y = batch.clone()
        batch = batch.reshape(bz, 3, self.segment, self.segment_length, batch.size(-1)) # (bz, 3, segment, segment_length, feature_dim)

        x = batch.reshape(bz*3*self.segment, -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        decoded = decoded.reshape(bz, 3, self.segment*self.segment_length, -1)
        return y, decoded

    def training_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        reconstruction_loss = F.mse_loss(y, y_hat, reduction='mean')
        
        y_vel = y[:, :, 1:, :] - y[:, :, :-1, :]
        y_hat_vel = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]
        velocity_loss = F.mse_loss(y_vel, y_hat_vel, reduction='mean')
        loss = reconstruction_loss + self.alpha * velocity_loss

        self.train_losses.append(loss)
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log('train_loss', torch.stack(self.train_losses).mean(), on_epoch=True, sync_dist=True)
        self.train_losses.clear()

    def on_validation_start(self):
        self.val_losses = []

    def validation_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.val_losses.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val_loss', torch.stack(self.val_losses).mean(), on_epoch=True, sync_dist=True)
        self.val_losses.clear()

    def on_test_start(self):
        self.test_losses = []

        result_dir = f'./result_autoencoder4/{self.task}/hidden{self.hidden_size}_lr{self.lr}_wd{self.weight_decay}_alpha{self.alpha}'
        model_dir = f'./pretrained4/{self.task}/hidden{self.hidden_size}_lr{self.lr}_wd{self.weight_decay}_alpha{self.alpha}'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.result_dir = result_dir
        self.model_dir = model_dir
        
    def test_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.test_losses.append(loss)
        if batch_idx >=  self.trainer.num_test_batches[0] - 2:
            if self.trainer.global_rank == 0:
                y = y[:, :, -self.segment_length:, :] 
                y_hat = y_hat[:, :, -self.segment_length:, :] 
                file_name = f'{self.result_dir}/{batch_idx}'
                visualize(self.task, self.normalizer, y, y_hat, file_name)
            

    
    def on_test_epoch_end(self):
        fps = 15 if self.task == 'pose' else 30
        for filename in os.listdir(self.result_dir):
            if filename.endswith('.mp4'):
                self.logger.experiment.log({f'{self.task}_video': wandb.Video(os.path.join(self.result_dir, filename), fps=fps, format="mp4")})

        avg_loss = torch.stack(self.test_losses).mean()
        self.log(f'lr{self.lr}_hidden{self.hidden_size}_wd{self.weight_decay}_{self.task}_test_loss', avg_loss) 
        self.test_losses.clear()
        
        torch.save(self.encoder.state_dict(), f"{self.model_dir}/encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.model_dir}/decoder.pth")


def main():
    wandb_logger = WandbLogger(project="masked-social-signals")
    hparams = wandb.config

    seed_everything(hparams.seed)

    torch.cuda.empty_cache()

    model = AutoEncoder(task=hparams.task,
                        segment=hparams.segment,
                        hidden_sizes=hparams.hidden_sizes,
                        alpha=hparams.alpha,
                        lr=hparams.lr,
                        weight_decay=hparams.weight_decay,
                        #warmup_ratio=hparams.warmup_ratio,
                        batch_size=hparams.batch_size)
    
    print(f'\nGrid Search on {hparams.model} model hidden_sizes={hparams.hidden_sizes} lr={hparams.lr} weight_decay={hparams.weight_decay} alpha={hparams.alpha}\n')

    name = f'{hparams.model}_lr{hparams.lr}_hidden{hparams.hidden_sizes}_wd{hparams.weight_decay}_alpha{hparams.alpha}'
    wandb_logger.experiment.name = name

    checkpoint_path = f'./checkpoints_autoencoder/{hparams.model}/lr{hparams.lr}_hidden{hparams.hidden_sizes}_wd{hparams.weight_decay}_alpha{hparams.alpha}'
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
                         )
    trainer.fit(model)
    
    best_model_path = checkpoint_callback.best_model_path
    best_model = AutoEncoder.load_from_checkpoint(best_model_path)
    trainer.test(best_model)

        
if __name__ == '__main__':
    with open('cfgs/autoencoder.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="masked-social-signals")
    wandb.agent(sweep_id, function=main)


