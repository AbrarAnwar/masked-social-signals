import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR
from lightning import LightningModule

from utils.dataset import get_loaders
from utils.visualize import *
from utils.embeddings import *
from utils.normalize import *
from utils.utils import *
from models.autoencoder import *


class LSTMModel(LightningModule):
    FEATURES = {'word':768, 'headpose':2, 'gaze':2, 'pose':26}

    def __init__(self, 
                 hidden_size, 
                 segment, 
                 task, 
                 multi_task, 
                 pretrained, 
                 frozen,
                 lr, 
                 weight_decay, 
                 warmup_ratio,
                 alpha,
                 batch_size):
        
        super().__init__()

        self.save_hyperparameters()

        self.task = task
        self.feature_dim = self.FEATURES[task]
        self.hidden_size = hidden_size
        self.segment = segment
        self.batch_length = 1080
        self.segment_length = int(self.batch_length / self.segment)
        self.multi_task = multi_task

        self.pretrained = pretrained
        self.frozen = frozen

        if self.multi_task:
            self.processor = BERTProcessor()
            self.task_list = ['headpose', 'gaze', 'pose', 'word']
            self.encoder = nn.ModuleList([Encoder(self.FEATURES['headpose'] * self.segment_length, [128, 64]),
                                    Encoder(self.FEATURES['gaze'] * self.segment_length, [128, 64]),
                                    Encoder(self.FEATURES['pose'] * self.segment_length // 2, [768, self.hidden_size]), # fps 15
                                    Encoder(self.FEATURES['word'] * self.segment_length, [128, 64])])

            self.decoder = nn.ModuleList([Decoder(self.FEATURES['headpose'] * self.segment_length, [64, 128]),
                                            Decoder(self.FEATURES['gaze'] * self.segment_length, [64, 128]),
                                            Decoder(self.FEATURES['pose'] * self.segment_length // 2, [self.hidden_size, 768])])

            if self.pretrained:
                #TODO
                for task_idx, task in enumerate(self.task_list[:-2]):
                    self.encoder[task_idx].load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained/{task}/hidden[128, 64]_lr0.0003_wd1e-05/encoder.pth"))
                    self.decoder[task_idx].load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained/{task}/hidden[128, 64]_lr0.0003_wd1e-05/decoder.pth"))
            
                # load pose encoder
                self.encoder[2].load_state_dict(torch.load("/home/tangyimi/masked_mine/pretrained3/pose/hidden[768, 256]_lr0.0003_wd1e-05_alpha0/encoder.pth"))
                self.decoder[2].load_state_dict(torch.load("/home/tangyimi/masked_mine/pretrained3/pose/hidden[768, 256]_lr0.0003_wd1e-05_alpha0/decoder.pth"))
                
                if self.frozen:
                    for encoder, decoder in zip(self.encoder, self.decoder):
                        freeze(encoder)
                        freeze(decoder)

            
            self.lstm = nn.LSTM(input_size=448, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
            
        else:       
            if task == 'pose':
                self.encoder = Encoder(self.feature_dim * self.segment_length // 2, [768, self.hidden_size])
                self.decoder = Decoder(self.feature_dim * self.segment_length // 2, [self.hidden_size, 768])
                self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True)

                if self.pretrained:
                    self.encoder.load_state_dict(torch.load("/home/tangyimi/masked_mine/pretrained3/pose/hidden[768, 256]_lr0.0003_wd1e-05_alpha0/encoder.pth"))
                    self.decoder.load_state_dict(torch.load("/home/tangyimi/masked_mine/pretrained3/pose/hidden[768, 256]_lr0.0003_wd1e-05_alpha0/decoder.pth"))
                    
                    if self.frozen:
                        freeze(self.encoder)
                        freeze(self.decoder)


            else:
                self.encoder = Encoder(self.feature_dim * self.segment_length, [128, 64])
                self.decoder = Decoder(self.feature_dim * self.segment_length, [64, 128])
                self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True)

                if self.pretrained:
                    self.encoder.load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained/{self.task}/hidden[128, 64]_lr0.0003_wd1e-05/encoder.pth"))
                    self.decoder.load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained/{self.task}/hidden[128, 64]_lr0.0003_wd1e-05/decoder.pth"))
                    
                    if self.frozen:
                        freeze(self.encoder)
                        freeze(self.decoder)
            

        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.alpha = alpha

        # load data
        self.batch_size = batch_size
        train_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v2', 30, training=True)

        split = int(0.8 * len(train_dataset))
        train_indices = list(range(split))
        val_indices = list(range(split, len(train_dataset)))

        self.train_dataset = Subset(train_dataset, train_indices)
        self.val_dataset = Subset(train_dataset, val_indices)
        self.test_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v2', 30, training=False)
        
        self.normalizer = Normalizer(self.train_dataloader())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
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

    def on_train_start(self):
        self.train_losses = []

    def forward_single_task(self, batch): 
        batch[self.task] = self.normalizer.minmax_normalize(batch[self.task], self.task)
        if self.task == 'pose':
            batch['pose'] = smoothing(batch['pose'], self.batch_length).to(self.device)
        segment_length = self.segment_length // 2 if self.task == 'pose' else self.segment_length

        batch = batch[self.task]
        bz = batch.size(0)
        batch = batch.reshape(bz, 3, self.segment, segment_length, batch.size(-1))
        train_segment = self.segment - 1

        x = batch[:, :, :train_segment, :, :] # (bz, 3, 5, 180, feature_dim)
        y = batch[:, :, -1, :, :].squeeze(2)

        x = x.reshape(bz*3*train_segment, -1)

        encode = self.encoder(x).view(-1, train_segment, self.hidden_size) 
        lstm_out, _ = self.lstm(encode)
        lstm_out = lstm_out[:, -1, :] 
        reconstructed = self.decoder(lstm_out).reshape(bz, 3, segment_length, -1)

        return y, reconstructed
                        
    def forward_multi_task(self, batch):
        batch['word'] = self.processor.get_embeddings(batch['word'])
        for task in self.task_list[:-1]:
            batch[task] = self.normalizer.minmax_normalize(batch[task], task)
        batch['pose'] = smoothing(batch['pose'], self.batch_length).to(self.device)

        encode_list = []
        bz = batch['gaze'].size(0)
        train_segment = self.segment - 1
        ys = []

        for task_idx, task in enumerate(self.task_list):
            current = batch[task]
            segment_length = self.segment_length // 2 if task == 'pose' else self.segment_length
            current = current.reshape(bz, 3, self.segment, segment_length, -1)
            x = current[:, :, :train_segment, :, :] # (bz, 3, 5, 180, feature_dim)
            x = x.reshape(bz*3*train_segment, -1)

            encode = self.encoder[task_idx](x).view(bz*3, train_segment, -1) # (bz*3, 11, hidden)
            
            encode_list.append(encode)

            if task != 'word':
                y = current[:, :, -1, :, :].squeeze(2) # (bz, 3, 180, feature_dim)
                ys.append(y) 
        
        encode = torch.cat(encode_list, dim=2)

        lstm_out, _ = self.lstm(encode)
        lstm_out = lstm_out[:, -1, :] # (bz*3, 1, 256)
        reconstructed = []
        for task_idx, task in enumerate(self.task_list[:-1]):
            if task == 'pose':
                y_hat = self.decoder[task_idx](lstm_out).reshape(bz, 3, self.segment_length//2, -1)
            else:
                y_hat = self.decoder[task_idx](lstm_out[:,:64]).reshape(bz, 3, self.segment_length, -1)
            reconstructed.append(y_hat)

        return ys, reconstructed

    def forward(self, batch):
        if self.multi_task:
            return self.forward_multi_task(batch) 
        return self.forward_single_task(batch)

    def training_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        if self.multi_task:
            losses = [] 
            for task_idx, task in enumerate(self.task_list[:-1]):
                current_y, current_y_hat = y[task_idx], y_hat[task_idx]
                reconstruct_loss = F.mse_loss(current_y, current_y_hat, reduction='mean')
                # calculate velocity loss
                y_vel = current_y[:, :, 1:, :] - current_y[:, :, :-1, :]
                y_hat_vel = current_y_hat[:, :, 1:, :] - current_y_hat[:, :, :-1, :]
                velocity = F.mse_loss(y_vel, y_hat_vel, reduction='mean')
                task_loss = reconstruct_loss + self.alpha * velocity

                self.log(f'train_loss/{task}', task_loss, on_epoch=True, sync_dist=True)
                losses.append(task_loss)
            loss = torch.stack(losses).mean()
            self.train_losses.append(losses)
        else:
            reconstruct_loss = F.mse_loss(y_hat, y, reduction='mean')
            # calculate velocity loss
            y_vel = y[:, :, 1:, :] - y[:, :, :-1, :]
            y_hat_vel = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]
            velocity_loss = F.mse_loss(y_vel, y_hat_vel, reduction='mean')
            loss = reconstruct_loss + velocity_loss
            self.train_losses.append(loss) 
            self.log(f'train_loss/{self.task}', loss, on_epoch=True, sync_dist=True)
        return loss
    
    def on_training_epoch_end(self):
        if self.multi_task:
            for task_idx, task in enumerate(self.task_list[:-1]):
                self.log(f'train_loss/{task}', 
                         torch.stack([loss[task_idx] for loss in self.train_losses]).mean(), 
                         on_epoch=True, 
                         sync_dist=True)
        else:
            self.log(f'train_loss/{self.task}', torch.stack(self.train_losses).mean(), on_epoch=True, sync_dist=True)
        self.train_losses.clear()

    def on_validation_start(self):
        self.val_losses = []

    def validation_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        if self.multi_task:
            losses = []
            for task_idx, _ in enumerate(self.task_list[:-1]):
                sub_loss = F.mse_loss(y_hat[task_idx], y[task_idx], reduction='mean')
                losses.append(sub_loss)
            loss = torch.stack(losses).mean()
            self.val_losses.append(losses)
        else:
            loss = F.mse_loss(y_hat, y, reduction='mean')
            self.val_losses.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        if self.multi_task:
            for task_idx, task in enumerate(self.task_list[:-1]):
                self.log(f'val_loss/{task}', 
                         torch.stack([loss[task_idx] for loss in self.val_losses]).mean(), 
                         on_epoch=True, 
                         sync_dist=True)
            average_loss = torch.tensor(self.val_losses).mean()
            self.log('val_loss', average_loss, on_epoch=True, sync_dist=True)
        else:
            self.log(f'val_loss', torch.stack(self.val_losses).mean(), on_epoch=True, sync_dist=True)
        self.val_losses.clear()
    
    # TODO: folder naming
    def on_test_start(self):
        root_dir = 'multi_task_result3'
        if self.multi_task:
            result_dir = f'./{root_dir}/lstm/multi_lr{self.lr}_frozen{self.frozen}'
            for i in self.task_list[:-1]:
                if not os.path.exists(f'{result_dir}/{i}'):
                    os.makedirs(f'{result_dir}/{i}')
        else:
            result_dir = f'./{root_dir}/lstm/single_lr{self.lr}_frozen{self.frozen}/{self.task}'

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

        self.result_dir = result_dir
        self.test_losses = [[] for _ in range(len(self.task_list[:-1]))] if self.multi_task else []
    
    def test_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        if self.multi_task:
            for task_idx, task in enumerate(self.task_list[:-1]):
                current_y = y[task_idx]
                current_y_hat = y_hat[task_idx]
                current_loss = F.mse_loss(current_y_hat, current_y, reduction='mean')
                self.test_losses[task_idx].append(current_loss)

                if batch_idx >= self.trainer.num_test_batches[0] - 2:
                    if self.trainer.global_rank == 0:
                        file_name = f'{self.result_dir}/{task}/{batch_idx}'
                        visualize(task, self.normalizer, current_y, current_y_hat, file_name)
        else:
            loss = F.mse_loss(y_hat, y, reduction='mean')
            self.test_losses.append(loss)

            if batch_idx >= self.trainer.num_test_batches[0] - 3:
                if self.trainer.global_rank == 0:
                    file_name = f'{self.result_dir}/{batch_idx}'
                    visualize(self.task, self.normalizer, y, y_hat, file_name)
    
    # calculate the total test reconstruction loss
    def on_test_epoch_end(self):
        if self.multi_task:
            loss_name = f'lstm_{self.task}_test_loss'
            for task_idx, task in enumerate(self.task_list[:-1]):
                avg_loss = torch.stack(self.test_losses[task_idx]).mean()
                self.log(f'transformer_multi_{task}_test_loss', avg_loss)

                fps = 15 if task == 'pose' else 30
                for filename in os.listdir(f'{self.result_dir}/{task}'):
                    if filename.endswith(".mp4"):
                        self.logger.experiment.log({f'video/{task}': wandb.Video(os.path.join(f'{self.result_dir}/{task}', filename), fps=fps, format="mp4")})
        else:
            loss_name = f'lstm_{self.task}_single_test_loss'
            avg_loss = torch.stack(self.test_losses).mean()
            self.log(loss_name, avg_loss)
            fps = 15 if self.task == 'pose' else 30
            for filename in os.listdir(f'{self.result_dir}'):
                if filename.endswith(".mp4"):
                    self.logger.experiment.log({f'video/{self.task}': wandb.Video(os.path.join(f'{self.result_dir}', filename), fps=fps, format="mp4")})
        self.test_losses.clear()

        
if __name__ == '__main__':
    model = LSTMModel(hidden_size=256,
                        segment=12,
                        task='pose',
                        multi_task=True,
                        pretrained=True,
                        frozen=True,
                        lr=0.0003,
                        weight_decay=1e-5,
                        warmup_ratio=0.1,
                        alpha = 0.5,
                        batch_size=16)
    wandb_logger = WandbLogger(project="sample")
    
    trainer = pl.Trainer(max_epochs=1, strategy=DDPStrategy(find_unused_parameters=True), logger=wandb_logger, num_sanity_val_steps=0)
    trainer.fit(model)
    trainer.test(model)



