import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR
import os
import random

from lightning import LightningModule
from lightning.pytorch.strategies import DDPStrategy
import lightning.pytorch as pl
import transformers

from models.autoencoder import *
from models.gpt2 import GPT2Model
from utils.dataset import *
from utils.visualize import *
from utils.embeddings import *
from utils.normalize import *

# (Return_1, state_1, action_1, Return_2, state_2, ...)
class MaskTransformer(LightningModule):
    FEATURES = {'word':768, 'headpose':2, 'gaze':2, 'pose':26}

    def __init__(
            self,
            hidden_size,
            segment,
            task,
            multi_task,
            mask_ratio,
            mask_strategy,
            eval_type,
            pretrained,
            lr,
            weight_decay,
            warmup_steps,
            batch_size,
            **kwargs
    ):
        super().__init__()
        assert multi_task or task is not None, 'task should be specified in single task setting'
        #assert eval_type==2 or task is not None, 'task should be specified in single task setting'
        
        self.task = task
        self.hidden_size = hidden_size
        self.segment = segment
        self.batch_length = 1080
        self.segment_length = int(self.batch_length / self.segment)
        self.multi_task = multi_task
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy # 1: completely random, 2: random one within each timestep
        self.eval_type = eval_type # in multi task setting, 1 = mask one task at the last timestep, 2 = mask certain features for all samples
        self.pretrained = pretrained

        self.processors = BERTProcessor()
        self.embed_timestep = nn.Embedding(self.segment, self.hidden_size)
        self.embed_speaker = nn.Embedding(4, self.hidden_size)
        self.task_list = ['headpose', 'gaze', 'pose', 'word'] # status_speaker

        self.encoder = nn.ModuleList([Encoder(self.FEATURES['headpose'] * self.segment_length, [128, self.hidden_size]),
                                    Encoder(self.FEATURES['gaze'] * self.segment_length, [128, self.hidden_size]),
                                    Encoder(self.FEATURES['pose'] * self.segment_length // 2, [768, 256]), # fps 15
                                    Encoder(self.FEATURES['word'] * self.segment_length, [128, self.hidden_size])])
        self.pose_projection_encoder = Encoder(256, [self.hidden_size], activation=False)

        # TODO: load pose autoencoder seperately and give pose a seperate projection encoder and decoder
        if self.pretrained:
            for task_idx, task in enumerate(self.task_list[:-2]):
                self.encoder[task_idx].load_state_dict(torch.load(f"./pretrained/{task}/hidden[128, 64]_lr0.0003_wd1e-05/encoder.pth"))
                self.encoder[task_idx].requires_grad_(False)
            
            # load pose encoder
            self.encoder[2].load_state_dict(torch.load(f"./pretrained2_15fps/pose/hidden[768, 256]_lr0.0003_wd1e-05/encoder.pth"))
            self.encoder[2].requires_grad_(False)

            
        if self.multi_task:
            self.decoder = nn.ModuleList([Decoder(self.FEATURES['headpose'] * self.segment_length, [self.hidden_size, 128]),
                                            Decoder(self.FEATURES['gaze'] * self.segment_length, [self.hidden_size, 128]),
                                            Decoder(self.FEATURES['pose'] * self.segment_length // 2, [256, 768])])
            self.pose_projection_decoder = Decoder(256, [self.hidden_size])

            if self.pretrained:
                for task_idx, task in enumerate(self.task_list[:-2]):
                    self.decoder[task_idx].load_state_dict(torch.load(f"./pretrained/{task}/hidden[128, 64]_lr0.0003_wd1e-05/decoder.pth"))
                    self.decoder[task_idx].requires_grad_(False)
                
                # load pose decoder
                self.decoder[2].load_state_dict(torch.load(f"./pretrained2_15fps/pose/hidden[768, 256]_lr0.0003_wd1e-05/decoder.pth"))
                self.decoder[2].requires_grad_(False)
                    

            feature_indices = torch.arange(len(self.task_list)*self.segment) % len(self.task_list)
            feature_mask = feature_indices < len(self.task_list) - 1
            self.feature_indices = torch.nonzero(feature_mask).squeeze() 

        else:
            if self.task == 'pose':
                self.pose_projection_decoder = Decoder(256, [self.hidden_size])
                self.decoder = Decoder(self.FEATURES[self.task] * self.segment_length // 2, [256, 768])

            else:
                self.decoder = Decoder(self.FEATURES[self.task] * self.segment_length, [self.hidden_size, 128])

            if self.pretrained:
                if self.task == 'pose':
                    self.decoder.load_state_dict(torch.load(f"./pretrained2_15fps/pose/hidden[768, 256]_lr0.0003_wd1e-05/decoder.pth"))
                    self.decoder.requires_grad_(False)
                else:
                    self.decoder.load_state_dict(torch.load(f"./pretrained/{self.task}/hidden[128, 64]_lr0.0003_wd1e-05/decoder.pth"))
                    self.decoder.requires_grad_(False)

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # load data
        self.batch_size = batch_size
        self.dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18') 
    
        train_size = int(0.8 * len(self.dataset))
        self.train_dataset = Subset(self.dataset, range(0, train_size))
        self.test_dataset = Subset(self.dataset, range(train_size, len(self.dataset)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    def configure_optimizers(self):
        betas = (0.9, 0.95)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=betas)
        lr_lambda = lambda epoch: self.lr_schedule(self.global_step)
        scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [scheduler]

    def lr_schedule(self, current_step):
        if current_step < self.warmup_steps:
            lr_mult = float(current_step) / float(max(1, self.warmup_steps))
        else:
            progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return self.lr * lr_mult
            
    def on_train_start(self):
        self.normalizer = Normalizer(self.train_dataloader())
        batches_per_epoch = len(self.train_dataloader())
        self.total_steps = batches_per_epoch * 20

    def forward_single_task(self, batch):
        # status speaker (16, 3, 1080) => (16, 1080) => embedding => (16, 1080, 64) => (16, 12, 90, 64)
        batch['word'] = self.processors.get_embeddings(batch['word'])
        encode_list = []
        bz = batch['gaze'].size(0)
        
        time = torch.arange(self.segment).expand(bz*3, -1).to(self.device)
        time_embeddings = self.embed_timestep(time) # (bz*3, 6, 64)

        for task_idx, task in enumerate(self.task_list):
            current = batch[task]
            if task == 'pose':
                current = current[:, :, ::2, :]
                segment_length = self.segment_length // 2
            else:
                segment_length = self.segment_length

            current = current.reshape(bz, 3, self.segment, segment_length, current.size(-1))
            current_reshaped = current.reshape(bz*3*self.segment, -1) # (bz*3*6, 180, 2)
            
            encode = self.encoder[task_idx](current_reshaped).view(bz*3, self.segment, -1) # (bz*3, 6, 64)

            if task == 'pose':
                encode = self.pose_projection_encoder(encode).view(bz*3, self.segment, self.hidden_size)

            if task == self.task:
                MASK = torch.zeros(1, self.hidden_size)
                encode[:, -1, :] = MASK 
                y = current.clone()[:, :, -1, :, :].squeeze(2) # (bz, 3, 180, feature_dim)

            encode = encode + time_embeddings
            encode_list.append(encode)

        # it will make the input as (headpose1, gaze1, pose1, word1, headpose2, gaze2, pose2, word2, ...) (bz*3, 24, 64)
        stacked_inputs = torch.stack(encode_list, dim=1).permute(0, 2, 1, 3).reshape(bz*3, len(self.task_list)*self.segment, -1)

        # (headpose_p1_t1, gaze_p1_t1, pose_p1_t1, word_p1_t1, headpose_p2_t1, gaze_p2_t1, pose_p2_t1, wordp2_t1, ...)
        stacked_inputs = stacked_inputs.view(bz, 3, -1, self.hidden_size).permute(0, 2, 1, 3).reshape(bz, -1, self.hidden_size) # (bz, 24*3, hidden_size)

        outputs = self.transformer(inputs_embeds=stacked_inputs)['last_hidden_state'] # (bz, 24*3, 64)

        outputs = outputs.view(bz, -1, 3, self.hidden_size).permute(0, 2, 1, 3).reshape(bz*3, -1, self.hidden_size) # (bz*3, 24, 64)

        task_index = self.task_list.index(self.task) - len(self.task_list)

        prediction = outputs[:, task_index, :].squeeze(1) # (bz*3, 64)

        if self.task == 'pose':
            prediction = self.pose_projection_decoder(prediction)
            segment_length = self.segment_length // 2
        else:
            segment_length = self.segment_length
        reconstructed = self.decoder(prediction).reshape(bz, 3, segment_length, -1)

        return y, reconstructed
    
    def forward_multi_task(self, batch):
        batch['word'] = self.processors.get_embeddings(batch['word'])
        encode_list = []
        original = []
        bz = batch['gaze'].size(0)
        time = torch.arange(self.segment).expand(bz*3, -1).to(self.device)
        time_embeddings = self.embed_timestep(time) 
        
        for task_idx, task in enumerate(self.task_list):
            current = batch[task]
            if task == 'pose':
                current = current[:, :, ::2, :] # 15fps
                segment_length = self.segment_length // 2
            else:
                segment_length = self.segment_length

            current = current.reshape(bz, 3, self.segment, segment_length, current.size(-1)) # (bz, 3, 6, 180, feature_dim)
            original.append(current.clone()) 

            current_reshaped = current.reshape(bz*3*self.segment, -1) # (bz*3*6, 180, 2)
            
            encode = self.encoder[task_idx](current_reshaped).view(bz*3, self.segment, -1) # (bz*3, 6, hidden_size)

            if task == 'pose':
                encode = self.pose_projection_encoder(encode).view(bz*3, self.segment, self.hidden_size) # (bz*3, 6, hidden_size)

            encode_list.append(encode)
        
        # it will make the input as (headpose1, gaze1, pose1, word1, headpose2, gaze2, pose2, word2, ...) (bz, 24, 64)
        stacked_inputs = torch.stack(encode_list, dim=1).permute(0, 2, 1, 3).reshape(bz*3, len(self.task_list)*self.segment, -1) # (bz*3, 24, 64)
        time_embeddings_repeated = torch.repeat_interleave(time_embeddings, len(self.task_list), dim=1) # (bz*3, 24, 64)

        # create mask indices shape like (bz, m) this means it has different mask indices for each sequence in this batch
        if self.training:
            if self.mask_strategy == 1:
                num_to_mask = int(self.mask_ratio * self.feature_indices.numel())
                mask_indices = torch.stack([self.feature_indices[torch.randperm(self.feature_indices.numel())[:num_to_mask]] for _ in range(bz)], dim=0) # (bz, m)
                mask_indices = mask_indices.repeat(3,1) # (bz*3, m)

            elif self.mask_strategy == 2:
                reshape_feature_indices = self.feature_indices.reshape(self.segment, -1).unsqueeze(0).repeat(bz,1,1) # 
                random_indices = torch.randint(0, len(self.task_list)-1, size=(bz, reshape_feature_indices.size(1))).unsqueeze(-1)
                mask_indices = torch.gather(reshape_feature_indices, dim=-1, index=random_indices).squeeze(-1)
                mask_indices = mask_indices.repeat(3,1) # (bz*3, m)
            else:
                raise NotImplementedError('Mask strategy should be 1 or 2')
        else:
            mask_indices = self.certain_mask_indices[:bz]
            mask_indices = mask_indices.repeat(3,1) 
        
        
        full_mask = torch.zeros_like(stacked_inputs, dtype=torch.bool)
        full_mask[torch.arange(bz*3)[:, None], mask_indices, :] = True
        stacked_inputs[full_mask] = 0

        stacked_inputs = stacked_inputs + time_embeddings_repeated 

        # it will make the input as (headpose_p1_t1, gaze_p1_t1, pose_p1_t1, word_p1_t1, headpose_p2_t1, gaze_p2_t1, pose_p2_t1, wordp2_t1, ...) 
        stacked_inputs = stacked_inputs.view(bz, 3, -1, self.hidden_size).permute(0, 2, 1, 3).reshape(bz, -1, self.hidden_size) # (bz, 24*3, hidden_size)

        output = self.transformer(inputs_embeds=stacked_inputs)['last_hidden_state'] # (bz, 24*3, 64)

        output = output.view(bz, -1, 3, self.hidden_size).permute(0, 2, 1, 3).reshape(bz*3, -1, self.hidden_size) # (bz*3, 24, 64)

        ys = []
        y_hats = []

        batch_indices = torch.arange(bz*3).view(-1, 1).expand(-1, mask_indices.size(1))

        mask_task_flat = (mask_indices % len(self.task_list)).reshape(-1)
        batch_indices_flat = batch_indices.reshape(-1)
        mask_indices_flat = mask_indices.reshape(-1)

        # Create indices for each task. 
        # Each has shape of (k, 2) where k is the number of masked features. 2 means (batch_index, mask_index in this sequence)
        # Assumption: each task has at least one masked position. (Thus, did not handle no masked position case)
        for task_idx, task in enumerate(self.task_list[:-1]):
            current_indices = torch.stack((batch_indices_flat[mask_task_flat==task_idx], mask_indices_flat[mask_task_flat==task_idx]), dim=1) # (k, 2)

            current_output = output[current_indices[:, 0], current_indices[:, 1], :] # (k, hidden_size)
            
            if task == 'pose':
                current_output = self.pose_projection_decoder(current_output) # (k, 256)
                segment_length = self.segment_length // 2
            else:
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

        if self.multi_task and (self.eval_type == 2 or self.training):
            return self.forward_multi_task(batch)
        return self.forward_single_task(batch)
    
    def training_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        if self.multi_task:
            losses = []
            for task_idx, task in enumerate(self.task_list[:-1]):
                sub_loss = F.mse_loss(y_hat[task_idx], y[task_idx], reduction='mean')
                self.log(f'transformer_{task}_train_loss', sub_loss)
                losses.append(sub_loss)
            loss = torch.stack(losses).mean()
        else:
            loss = F.mse_loss(y_hat, y, reduction='mean')
        self.log('train_loss', loss)
        return loss
    
    def on_test_start(self):
        self.normalizer = Normalizer(self.train_dataloader())
        root_dir = 'result7'
        self.test_losses = []

        if self.multi_task:
            if self.pretrained:
                result_dir = f'./{root_dir}/transformer/multi/mask{self.mask_strategy}_eval{self.eval_type}_segment{self.segment}_pretrained'
            else:
                result_dir = f'./{root_dir}/transformer/multi/mask{self.mask_strategy}_eval{self.eval_type}_segment{self.segment}'

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
                result_dir = f'./{root_dir}/transformer/single/segment{self.segment}__pretrained/{self.task}'
            else:
                result_dir = f'./{root_dir}/transformer/single/segment{self.segment}/{self.task}'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.result_dir = result_dir
        self.total_batches = len(self.test_dataloader())

    # TODO: change fps
    def test_step(self, batch, batch_idx):
        output = 0
        y, y_hat = self.forward(batch)
        if self.multi_task and self.eval_type == 2:
            for task_idx, task in enumerate(self.task_list[:-1]):
                current_y = y[task_idx]
                current_y_hat = y_hat[task_idx]
                current_loss = F.mse_loss(current_y_hat, current_y, reduction='mean')
                self.test_losses[task_idx].append(current_loss)

                if batch_idx >= self.total_batches - 3:
                    fps = 15 if task == 'pose' else 30
                    current_y = self.normalizer.minmax_denormalize(current_y, task)
                    current_y_hat = self.normalizer.minmax_denormalize(current_y_hat, task)

                    videos_prediction = construct_batch_video(current_y_hat, task=task)
                    videos_inference = construct_batch_video(current_y, task=task, color=(0,0,255))

                    result_videos = np.concatenate([videos_prediction, videos_inference], axis=2)

                    for i in range(current_y.size(0)):
                        write_video(result_videos[i], f'{self.result_dir}/{task}/{batch_idx}_{output}.mp4', fps)
                        output += 1
        else:
            loss = F.mse_loss(y_hat, y, reduction='mean')
            self.test_losses.append(loss)

            if batch_idx >= self.total_batches - 3:
                fps = 15 if self.task == 'pose' else 30
                current_y = self.normalizer.minmax_denormalize(y, self.task)
                current_y_hat = self.normalizer.minmax_denormalize(y_hat, self.task)

                videos_prediction = construct_batch_video(current_y_hat, task=self.task)
                videos_inference = construct_batch_video(current_y, task=self.task, color=(0,0,255))

                result_videos = np.concatenate([videos_prediction, videos_inference], axis=2)

                for i in range(batch[self.task].shape[0]):
                    write_video(result_videos[i], f'{self.result_dir}/{batch_idx}_{output}.mp4', fps)
                    output += 1


    def on_test_epoch_end(self):
        if self.multi_task:
            if self.eval_type == 1:
                avg_loss = torch.stack(self.test_losses).mean()
                loss_name = f'transformer_multi_{self.mask_strategy}_{self.task}_test_loss'
                self.log(loss_name, avg_loss) 

            elif self.eval_type == 2:
                loss_name = f'transformer_{self.mask_strategy}_test_loss'
                for task_idx, task in enumerate(self.task_list[:-1]):
                    avg_loss = torch.stack(self.test_losses[task_idx]).mean()
                    self.log(f'transformer_multi_{self.mask_strategy}_{task}_test_loss', avg_loss)
        else:
            avg_loss = torch.stack(self.test_losses).mean()
            loss_name = f'transformer_single_{self.task}_test_loss'
            self.log(loss_name, avg_loss)
            # write the loss to txt file
            with open(f'{self.result_dir}/{avg_loss.item()}.txt', 'w') as f:
                f.write(str(avg_loss.item()))
        self.test_losses.clear()


def main():
    model = MaskTransformer(hidden_size=64,
                            segment=12,
                            task='gaze',
                            multi_task=False,
                            mask_ratio=1/3,
                            mask_strategy=1,
                            eval_type=1,
                            pretrained=True,
                            lr=3e-4,
                            weight_decay=1e-5,
                            warmup_steps=100,
                            batch_size=16,
                            n_layer=6,
                            n_head=8,
                            n_inner=64*4,
                            activation_function='relu',
                            n_positions=512,
                            n_ctx=512,
                            resid_pdrop=0.1,
                            attn_pdrop=0.1)
    
    trainer = pl.Trainer(max_epochs=1, strategy=DDPStrategy(find_unused_parameters=True), logger=True)
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    main()
