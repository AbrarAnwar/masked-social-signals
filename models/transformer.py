import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR
import os

from lightning import LightningModule
from lightning.pytorch.strategies import DDPStrategy
import lightning.pytorch as pl
import transformers
from lightning.pytorch.loggers import WandbLogger

from models.autoencoder import *
from models.gpt2 import GPT2Model
from utils.dataset import *
from utils.visualize import *
from utils.embeddings import *
from utils.normalize import *
import utils
import wandb


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

class MaskTransformer(LightningModule):
    FEATURES = {'word':768, 'headpose':2, 'gaze':2, 'pose':26}

    def __init__(
            self,
            hidden_size,
            segment,
            task,
            frozen,
            multi_task,
            mask_ratio,
            eval_type,
            pretrained,
            feature_filling,
            lr,
            weight_decay,
            warmup_ratio,
            batch_size,
            alpha,
            **kwargs
    ):
        super().__init__()
        assert multi_task or task is not None, 'task should be specified in single task setting'
        #assert eval_type==2 or task is not None, 'task should be specified in single task setting'
        self.save_hyperparameters()

        self.task = task
        self.hidden_size = hidden_size
        self.frozen = frozen
        self.segment = segment
        self.batch_length = 1080
        self.segment_length = int(self.batch_length / self.segment)
        self.multi_task = multi_task
        self.mask_ratio = mask_ratio
        self.eval_type = eval_type # in multi task setting, 1 = mask one task at the last timestep, 2 = mask certain features for all samples
        self.pretrained = pretrained
        self.feature_filling = feature_filling

        self.processors = BERTProcessor()
        self.embed_timestep = nn.Embedding(self.segment, self.hidden_size)
        self.embed_speaker = nn.Embedding(4, self.hidden_size)
        self.task_list = ['headpose', 'gaze', 'pose', 'word'] # status_speaker

        self.encoder = nn.ModuleList([Encoder(self.FEATURES['headpose'] * self.segment_length, [128, 64]),
                                    Encoder(self.FEATURES['gaze'] * self.segment_length, [128, 64]),
                                    Encoder(self.FEATURES['pose'] * self.segment_length // 2, [768, self.hidden_size]), # fps 15
                                    Encoder(self.FEATURES['word'] * self.segment_length, [128, self.hidden_size])])

        if self.pretrained:
            for task_idx, task in enumerate(self.task_list[:-2]):
                self.encoder[task_idx].load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained/{task}/hidden[128, 64]_lr0.0003_wd1e-05/encoder.pth"))
            
            # load pose encoder
            self.encoder[2].load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained3/pose/hidden[768, 256]_lr0.0003_wd1e-05_alpha0/encoder.pth"))

        if self.frozen:
            for encoder in self.encoder:
                freeze(encoder)

        if self.multi_task:
            self.decoder = nn.ModuleList([Decoder(self.FEATURES['headpose'] * self.segment_length, [64, 128]),
                                            Decoder(self.FEATURES['gaze'] * self.segment_length, [64, 128]),
                                            Decoder(self.FEATURES['pose'] * self.segment_length // 2, [self.hidden_size, 768])])

            if self.pretrained:
                for task_idx, task in enumerate(self.task_list[:-2]):
                    self.decoder[task_idx].load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained/{task}/hidden[128, 64]_lr0.0003_wd1e-05/decoder.pth"))
                      
                # load pose decoder
                self.decoder[2].load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained3/pose/hidden[768, 256]_lr0.0003_wd1e-05_alpha0/decoder.pth"))
                
            if self.frozen:
                for decoder in self.decoder:
                    freeze(decoder)
                    
            # for later random masking
            feature_indices = torch.arange(len(self.task_list)*self.segment) % len(self.task_list)
            feature_mask = feature_indices < len(self.task_list) - 1
            self.feature_indices = torch.nonzero(feature_mask).squeeze()

        else:
            if self.task == 'pose':
                self.decoder = Decoder(self.FEATURES[self.task] * self.segment_length // 2, [self.hidden_size, 768])

            else:
                self.decoder = Decoder(self.FEATURES[self.task] * self.segment_length, [self.hidden_size, 128])

            if self.pretrained:
                if self.task == 'pose':
                    self.decoder.load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained3/pose/hidden[768, 256]_lr0.0003_wd1e-05_alpha0/decoder.pth"))
                else:
                    self.decoder.load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained/{self.task}/hidden[128, 64]_lr0.0003_wd1e-05/decoder.pth"))
            
            if self.frozen:
                freeze(self.decoder)


        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio 
        self.alpha = alpha

        # load data
        self.batch_size = batch_size
        #dataset = MultiDataset('/data/tangyimi/batch_window36_stride18') 
        dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18')

        self.train_dataset, self.val_dataset, self.test_dataset = \
                utils.random_split(dataset, [.8, .1, .1], generator=torch.Generator().manual_seed(42)) 
        
        self.normalizer = Normalizer(self.train_dataloader())
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    
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
        self.normalizer = Normalizer(self.train_dataloader())
        self.train_losses = []
        self.testing = False


    def forward_multi_task(self, batch):
        batch['word'] = self.processors.get_embeddings(batch['word'])
        batch['pose'] = smoothing(batch['pose'], self.batch_length).to(self.device)
        bz = batch['gaze'].size(0)
        time = torch.arange(self.segment).expand(bz*3, -1).to(self.device)
        time_embeddings = self.embed_timestep(time) 

        encode_list = []
        original = []
        
        for task_idx, task in enumerate(self.task_list):
            current = batch[task]
            if task == 'pose':
                # current = current[:, :, ::2, :] # 15fps
                segment_length = self.segment_length // 2
            else:
                segment_length = self.segment_length

            current = current.reshape(bz, 3, self.segment, segment_length, current.size(-1)) # (bz, 3, 6, 180, feature_dim)
            original.append(current.clone()) 

            current_reshaped = current.reshape(bz*3*self.segment, -1) # (bz*3*6, 180, 2)
            
            encode = self.encoder[task_idx](current_reshaped).view(bz*3, self.segment, -1) # (bz*3, 6, hidden_size)

            if task in {'headpose', 'gaze'}:
                
                if self.feature_filling == 'pad':
                    encode = F.pad(encode, (0, self.hidden_size - 64))
                elif self.feature_filling == 'repeat':
                    factor = self.hidden_size // 64
                    encode = torch.cat([encode]*factor, dim=2)
                else:
                    raise NotImplementedError('feature filling should be either pad or repeat')

            encode_list.append(encode)
        
        # it will make the input as (headpose1, gaze1, pose1, word1, headpose2, gaze2, pose2, word2, ...) (bz, 24, 64)
        stacked_inputs = torch.stack(encode_list, dim=1).permute(0, 2, 1, 3).reshape(bz*3, len(self.task_list)*self.segment, -1) # (bz*3, 24, 64)
        time_embeddings_repeated = torch.repeat_interleave(time_embeddings, len(self.task_list), dim=1) # (bz*3, 24, 64)
        
        stacked_inputs = stacked_inputs + time_embeddings_repeated 

        # it will make the input as (headpose_p1_t1, gaze_p1_t1, pose_p1_t1, word_p1_t1, headpose_p2_t1, gaze_p2_t1, pose_p2_t1, wordp2_t1, ...)
        stacked_inputs = stacked_inputs.view(bz, 3, -1, self.hidden_size).permute(0, 2, 1, 3).reshape(bz, -1, self.hidden_size) # (bz, 24*3, hidden_size)

        output = self.transformer(inputs_embeds=stacked_inputs)['last_hidden_state'] # (bz, 24*3, 64)

        output = output.view(bz, -1, 3, self.hidden_size).permute(0, 2, 1, 3).reshape(bz*3, -1, self.hidden_size) # (bz*3, 24, 64)

        if not self.testing:
            ys, y_hats = [], []
            for task_idx, task in enumerate(self.task_list[:-1]):
                
                task_output = output[:, task_idx::4, :]
                if task == 'pose':
                    segment_length = self.segment_length // 2   
                else:
                    task_output = task_output[:, :, :64]
                    segment_length = self.segment_length

                task_output_reshaped = task_output.reshape(bz*3*self.segment, -1)

                decode = self.decoder[task_idx](task_output_reshaped).reshape(bz, 3, self.segment*segment_length, -1) # (bz, 3, 6*180, feature_dim)
                y = original[task_idx].reshape(bz, 3, self.segment*segment_length, -1)

                y_hats.append(decode)
                ys.append(y)
        
        else:
            mask_indices = self.certain_mask_indices[:bz]
            mask_indices = mask_indices.repeat(3,1) 
            ys = []
            y_hats = []

            batch_indices = torch.arange(bz*3).view(-1, 1).expand(-1, mask_indices.size(1))

            mask_task_flat = (mask_indices % len(self.task_list)).reshape(-1)
            batch_indices_flat = batch_indices.reshape(-1)
            mask_indices_flat = mask_indices.reshape(-1)

            # Each has shape of (k, 2) where k is the number of masked features. 2 means (batch_index, mask_index in this sequence)
            # Assumption: each task has at least one masked position. (Thus, did not handle no masked position case)
            for task_idx, task in enumerate(self.task_list[:-1]):
                current_indices = torch.stack((batch_indices_flat[mask_task_flat==task_idx], mask_indices_flat[mask_task_flat==task_idx]), dim=1) # (k, 2)

                current_output = output[current_indices[:, 0], current_indices[:, 1], :] # (k, hidden_size)
                
                if task == 'pose':
                    segment_length = self.segment_length // 2
                else:
                    current_output = current_output[:, :64]
                    segment_length = self.segment_length
                
                current_reconstructed = self.decoder[task_idx](current_output)\
                                        .reshape(-1, 3, segment_length, self.FEATURES[task]) # (k, 3, 180, feature_dim)

                original_indices = current_indices.clone()
                original_indices[:, 1] //= len(self.task_list) # (k, 2)
                original_y_reshaped = original[task_idx].reshape(bz*3, self.segment, segment_length, -1)
                original_y = original_y_reshaped[original_indices[:, 0], original_indices[:, 1], :, :]\
                            .view(-1, 3, segment_length, self.FEATURES[task]) 
                
                y_hats.append(current_reconstructed)
                ys.append(original_y)
            

        return ys, y_hats
        

    def forward(self, batch):
        for task in self.task_list[:-1]:
            batch[task] = self.normalizer.minmax_normalize(batch[task], task)

        return self.forward_multi_task(batch)
    
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
            loss = F.mse_loss(y_hat, y, reduction='mean')
            self.train_losses.append(loss)
            self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss
    
    def on_training_epoch_end(self):
        if self.multi_task:
            for task_idx, task in enumerate(self.task_list[:-1]):
                self.log(f'train_loss/{task}', 
                         torch.stack([loss[task_idx] for loss in self.train_losses]).mean(), 
                         on_epoch=True, 
                         sync_dist=True)
        else:
            self.log('train_loss', torch.stack(self.train_losses).mean(), on_epoch=True, sync_dist=True)
    

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
            self.log('val_loss', torch.stack(self.val_losses).mean(), on_epoch=True, sync_dist=True)
    
    def on_test_start(self):
        #self.normalizer = Normalizer(self.train_dataloader())
        root_dir = 'multi_task_result3'
        self.test_losses = []
        self.testing = True

        if self.multi_task:
            if self.pretrained:
                result_dir = f'./{root_dir}/transformer/multi/eval{self.eval_type}_hidden{self.hidden_size}_lr{self.lr}_wd{self.weight_decay}_wr{self.warmup_ratio}_pretrained'
            else:
                result_dir = f'./{root_dir}/transformer/multi/eval{self.eval_type}_hidden{self.hidden_size}_lr{self.lr}_wd{self.weight_decay}_wr{self.warmup_ratio}'

            for i in self.task_list[:-1]:
                if not os.path.exists(f'{result_dir}/{i}'):
                    os.makedirs(f'{result_dir}/{i}')
            
            if self.eval_type == 1:
                result_dir = f'{result_dir}/{self.task}'
                self.decoder = self.decoder[self.task_list.index(self.task)]

            elif self.eval_type == 2:
                num_to_mask = int(self.mask_ratio * self.feature_indices.numel())
                mask_indices = torch.stack([self.feature_indices[torch.randperm(self.feature_indices.numel())[:num_to_mask]] for _ in range(self.batch_size)], dim=0) # (bz, m)
                self.certain_mask_indices = mask_indices
                self.test_losses = [[] for _ in range(len(self.task_list[:-1]))]

        else:
            if self.pretrained:
                result_dir = f'./{root_dir}/transformer/single/hidden{self.hidden_size}_lr{self.lr}_wd{self.weight_decay}_wr{self.warmup_ratio}_pretrained/{self.task}'
            else:
                result_dir = f'./{root_dir}/transformer/single/hidden{self.hidden_size}_lr{self.lr}_wd{self.weight_decay}_wr{self.warmup_ratio}/{self.task}'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.result_dir = result_dir


    def test_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        if self.multi_task and self.eval_type == 2:
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

            if batch_idx >= self.trainer.num_test_batches[0] - 2:
                if self.trainer.global_rank == 0:
                    file_name = f'{self.result_dir}/{batch_idx}'
                    visualize(self.task, self.normalizer, y, y_hat, file_name)


    def on_test_epoch_end(self):
        if self.multi_task:
            if self.eval_type == 1:
                avg_loss = torch.stack(self.test_losses).mean()
                loss_name = f'transformer_multi_{self.task}_test_loss'
                self.log(loss_name, avg_loss) 

            elif self.eval_type == 2:
                loss_name = f'transformer_test_loss'
                for task_idx, task in enumerate(self.task_list[:-1]):
                    avg_loss = torch.stack(self.test_losses[task_idx]).mean()
                    self.log(f'transformer_multi_{task}_test_loss', avg_loss)

                    fps = 15 if task == 'pose' else 30
                    for filename in os.listdir(f'{self.result_dir}/{task}'):
                        if filename.endswith(".mp4"):
                            self.logger.experiment.log({f'video/{task}': wandb.Video(os.path.join(f'{self.result_dir}/{task}', filename), fps=fps, format="mp4")})
        else:
            avg_loss = torch.stack(self.test_losses).mean()
            loss_name = f'transformer_single_{self.task}_test_loss'
            self.log(loss_name, avg_loss)
            # write the loss to txt file
            with open(f'{self.result_dir}/{avg_loss.item()}.txt', 'w') as f:
                f.write(str(avg_loss.item()))
        self.test_losses.clear()
        self.testing = False


def main():
    model = MaskTransformer(hidden_size=256, 
                            segment=12, 
                            task='pose', 
                            frozen=False, 
                            multi_task=True, 
                            mask_ratio=1/3, 
                            eval_type=2, 
                            pretrained=True, 
                            feature_filling='repeat', 
                            lr=3e-4, 
                            weight_decay=1e-5, 
                            warmup_ratio=0.1, 
                            batch_size=16, 
                            alpha=1,
                            n_layer=6,
                            n_head=8,
                            n_inner=256*4,
                            activation_function='relu',
                            n_ctx=144,
                            resid_pdrop=0.1,
                            attn_pdrop=0.1)
    
    wandb_logger = WandbLogger(project="sample")
    
    trainer = pl.Trainer(max_epochs=1, strategy=DDPStrategy(find_unused_parameters=True), logger=wandb_logger, num_sanity_val_steps=0,)
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    main()

