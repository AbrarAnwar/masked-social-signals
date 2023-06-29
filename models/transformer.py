from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os

from lightning import LightningModule
from lightning.pytorch.strategies import DDPStrategy
import lightning.pytorch as pl
import transformers

from models.base import Encoder, Decoder
from models.gpt2 import GPT2Model
from utils.dataset import *
from utils.visualize import *
from utils.embeddings import *

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
            lr,
            weight_decay,
            **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.segment = segment
        self.batch_length = 1080
        self.segment_length = int(self.batch_length / self.segment)
        self.multi_task = multi_task
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy # 1: completely random, 2: random one within each timestep

        # testing
        self.output = 0
        self.test_losses = []

        self.processors = BERTProcessor()
        self.embed_timestep = nn.Embedding(self.segment, self.hidden_size)
        self.task_list = ['headpose', 'gaze', 'pose', 'word']
        self.encoder = nn.ModuleList([Encoder(self.FEATURES['headpose'], self.segment_length, self.hidden_size),
                                        Encoder(self.FEATURES['gaze'], self.segment_length, self.hidden_size),
                                        Encoder(self.FEATURES['pose'], self.segment_length, self.hidden_size),
                                        Encoder(self.FEATURES['word'], self.segment_length, self.hidden_size)])

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        if not self.multi_task:
            self.task = task
            self.feature_dim = self.FEATURES[self.task]
            self.decoder = Decoder(self.hidden_size, self.segment_length, self.feature_dim)
        else:
            self.decoder = nn.ModuleList([Decoder(self.hidden_size, self.segment_length, self.FEATURES['headpose']),
                                        Decoder(self.hidden_size, self.segment_length, self.FEATURES['gaze']),
                                        Decoder(self.hidden_size, self.segment_length, self.FEATURES['pose'])])
            feature_indices = torch.arange(len(self.task_list)*self.segment) % len(self.task_list)
            feature_mask = feature_indices < len(self.task_list) - 1
            self.feature_indices = torch.nonzero(feature_mask).squeeze()  # [ 0,  1,  2,  4,  5,  6,  8,  9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]

        self.lr = lr
        self.weight_decay = weight_decay
            

    def forward_single_task(self, batch):
        batch['word'] = self.processors.get_embeddings(batch['word'])
        encode_list = []
        bz = batch['pose'].size(0)
        time = torch.arange(self.segment).expand(bz, -1).to(self.device)
        time_embeddings = self.embed_timestep(time) # (bz, 6, 64)

        for task_idx, task in enumerate(self.task_list):
            current = batch[task]
            current = current.reshape(bz, 3, self.segment, self.segment_length, current.size(-1))
            current_reshaped = current.permute(0, 2, 1, 3, 4).reshape(bz*self.segment, -1) # (bz*6, 180, 2)
            
            encode = self.encoder[task_idx](current_reshaped).view(bz, self.segment, -1) # (bz, 6, 64)

            if task == self.task:
                MASK = torch.zeros(1, self.hidden_size)
                # clone()
                encode[:, -1, :] = MASK 
                y = current.clone()[:, :, -1, :, :].squeeze(2) # (bz, 3, 180, feature_dim)

            encode = encode + time_embeddings
            encode_list.append(encode)

        # it will make the input as (headpose1, gaze1, pose1, word1, headpose2, gaze2, pose2, word2, ...) (bz, 24, 64)
        stacked_inputs = torch.stack(encode_list, dim=1).permute(0, 2, 1, 3).reshape(bz, len(self.task_list)*self.segment, -1)

        outputs = self.transformer(inputs_embeds=stacked_inputs)['last_hidden_state'] # (bz, 24, 64)

        mask_index = self.task_list.index(self.task) - 4

        prediction = outputs[:, mask_index, :].squeeze(1) # (bz, 64)

        reconstructed = self.decoder(prediction).reshape(bz, 3, self.segment_length, -1)

        return y, reconstructed
    
    def forward_multi_task(self, batch):
        batch['word'] = self.processors.get_embeddings(batch['word'])
        encode_list = []
        original = []
        bz = batch['pose'].size(0)
        time = torch.arange(self.segment).expand(bz, -1).to(self.device)
        time_embeddings = self.embed_timestep(time) # (bz, 6, 64)
        timestep_embedding_repeated = time_embeddings.repeat(1, len(self.task_list), 1)

        for task_idx, task in enumerate(self.task_list):
            current = batch[task]
            current = current.reshape(bz, 3, self.segment, self.segment_length, current.size(-1)) # (bz, 3, 6, 180, feature_dim)
            original.append(current.clone())

            current_reshaped = current.permute(0, 2, 1, 3, 4).reshape(bz*self.segment, -1) # (bz*6, 180, 2)
            
            encode = self.encoder[task_idx](current_reshaped).view(bz, self.segment, -1) # (bz, 6, 64)
            #encode = encode + time_embeddings
            encode_list.append(encode)
        
        stacked_inputs = torch.stack(encode_list, dim=1).permute(0, 2, 1, 3).reshape(bz, len(self.task_list)*self.segment, -1) # (bz, 24, 64)

        # create mask indices shape like (bz, m) this means it has different mask indices for each sequence in this batch
        
        if self.mask_strategy == 1:
            num_to_mask = int(self.mask_ratio * self.feature_indices.numel())
            mask_indices = self.feature_indices[torch.randperm(self.feature_indices.numel())[:num_to_mask]] # (m, ) m is the number of masked features

        elif self.mask_strategy == 2:
            reshape_feature_indices = self.feature_indices.reshape(self.segment, -1) 
            random_indices = torch.randint(0, len(self.task_list)-1, size=(reshape_feature_indices.size(0),))
            mask_indices = reshape_feature_indices[torch.arange(self.segment), random_indices]
        else:
            raise NotImplementedError('Mask strategy should be 1 or 2')

        # Create the full mask
        full_mask = torch.zeros(len(self.task_list)*self.segment, dtype=torch.bool) # 
        full_mask[mask_indices] = True
        full_mask = full_mask.unsqueeze(0).unsqueeze(2).repeat(bz, 1, self.hidden_size)

        #masked_data = stacked_inputs.clone()
        timestep_embedding_repeated = time_embeddings.repeat(1, len(self.task_list), 1) # 
        stacked_inputs[full_mask] = 0
        stacked_inputs += timestep_embedding_repeated

        output = self.transformer(inputs_embeds=stacked_inputs)['last_hidden_state'] # (bz, 24, 64)

        predictions = output[:, mask_indices, :] # (bz, m, 64) m is the number of masked features
        #print('Predictions shape:', predictions.shape)
        ys = []
        results = []
        result_task_index = []

        for i, mask_idx in enumerate(mask_indices):
            task_index = int(mask_idx % len(self.task_list))
            original_index = int(mask_idx // len(self.task_list))
            y = original[task_index][:, :, original_index, :, :].squeeze(2) # (bz, 3, 180, feature_dim)
            reconstructed = self.decoder[task_index](predictions[:, i, :].squeeze(1)).reshape(bz, 3, self.segment_length, -1) # (bz, 3, 180, feature_dim)

            ys.append(y)
            results.append(reconstructed)
            result_task_index.append(task_index)
        '''

        if self.mask_strategy == 1:
            num_to_mask = int(self.mask_ratio * self.feature_indices.numel())
            mask_indices = torch.stack([self.feature_indices[torch.randperm(self.feature_indices.numel())[:num_to_mask]] for _ in range(bz)], dim=0) # (bz, m)

        elif self.mask_strategy == 2:
            reshape_feature_indices = self.feature_indices.reshape(self.segment, -1).unsqueeze(0).repeat(bz,1,1)
            random_indices = torch.randint(0, len(self.task_list)-1, size=(bz, reshape_feature_indices.size(1))).unsqueeze(-1)
            mask_indices = torch.gather(reshape_feature_indices, dim=-1, index=random_indices).squeeze(-1)
        else:
            raise NotImplementedError('Mask strategy should be 1 or 2')
        
        full_mask = torch.zeros_like(stacked_inputs, dtype=torch.bool)
        full_mask[torch.arange(bz)[:, None], mask_indices, :] = True
        stacked_inputs[full_mask] = 0
        stacked_inputs += timestep_embedding_repeated

        output = self.transformer(inputs_embeds=stacked_inputs)['last_hidden_state'] # (bz, 24, 64)

        predictions = output[torch.arange(bz)[:, None], mask_indices, :] # (bz, m, 64) m is the number of masked features
        print('Predictions shape:', predictions.shape)
        print('Mask indices shape:', mask_indices.shape)
        print('mask indices:', mask_indices)
        print('mask indices % 4', mask_indices % 4)
        print('mask indices // 4', mask_indices // 4)
        return
        ys = []
        results = []
        result_task_index = []
        
        Predictions shape: torch.Size([4, 12, 64])
        Mask indices shape: torch.Size([4, 12])
        mask indices: tensor([[21, 41, 22, 20,  4, 14, 28, 40,  1, 18, 37,  8],
                [40, 34, 44, 33, 25, 16, 41, 46,  6,  4, 28, 20],
                [42, 41, 12,  2, 38, 24, 17, 45,  0, 28, 10, 46],
                [22, 46, 45, 42,  0, 12, 13, 30,  5,  8, 44,  1]])
        mask indices % 4 tensor([[1, 1, 2, 0, 0, 2, 0, 0, 1, 2, 1, 0],
                [0, 2, 0, 1, 1, 0, 1, 2, 2, 0, 0, 0],
                [2, 1, 0, 2, 2, 0, 1, 1, 0, 0, 2, 2],
                [2, 2, 1, 2, 0, 0, 1, 2, 1, 0, 0, 1]])
        mask indices // 4 tensor([[ 5, 10,  5,  5,  1,  3,  7, 10,  0,  4,  9,  2],
                [10,  8, 11,  8,  6,  4, 10, 11,  1,  1,  7,  5],
                [10, 10,  3,  0,  9,  6,  4, 11,  0,  7,  2, 11],
                [ 5, 11, 11, 10,  0,  3,  3,  7,  1,  2, 11,  0]])
        '''

        return torch.concatenate(ys, dim=-1), torch.concatenate(results, dim=-1), torch.tensor(result_task_index)

    def forward(self, batch):
        if self.multi_task:
            output = self.forward_multi_task(batch)
            return output[:2] if self.training else output
        return self.forward_single_task(batch)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean') 
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        if self.multi_task:
            result_dir = f'./transformer_result/multi/{self.mask_strategy}/{self.segment}/'
        else:
            result_dir = f'./transformer_result/single/{self.segment}/{self.task}'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if self.multi_task:
            for i in range(3):
                if not os.path.exists(f'{result_dir}/{self.task_list[i]}'):
                    os.makedirs(f'{result_dir}/{self.task_list[i]}')

            y, y_hat, result_task_index = self.forward(batch)
            previous_index = 0

            for task_index in result_task_index:
                task = self.task_list[task_index]
                current_index = previous_index + self.FEATURES[task]

                videos_prediction = construct_batch_video(y_hat[:, :, :, previous_index:current_index], task=task)
                videos_inference = construct_batch_video(y[:, :, :, previous_index:current_index], task=task, color=(0,0,255))
                previous_index = current_index
                result_videos = np.concatenate([videos_prediction, videos_inference], axis=2)

                for i in range(y.shape[0]):
                    write_video(result_videos[i], f'{result_dir}/{task}/{self.output}.mp4')
                    self.output += 1
        else:
            y, y_hat = self.forward(batch) # y (bz, 3, 180, feature_dim) y_hat (bz, 3, 180, feature_dim)
            videos_prediction = construct_batch_video(y_hat, task=self.task)
            videos_inference = construct_batch_video(y, task=self.task, color=(0,0,255))

            result_videos = np.concatenate([videos_prediction, videos_inference], axis=2)

            for i in range(batch[self.task].shape[0]):
                write_video(result_videos[i], f'{result_dir}/{self.output}.mp4')
                self.output += 1

        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.test_losses.append(loss)
        return loss


    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        if self.multi_task:
            loss_name = f'transformer_{self.mask_strategy}_test_loss'
        else:
            loss_name = f'transformer_{self.task}_single_test_loss'
        self.log(loss_name, avg_loss) 
        self.test_losses.clear()


def main():
    # load data
    single_task = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18') 
    
    train_size = int(0.8 * len(single_task))
    train_dataset = Subset(single_task, range(0, train_size))
    test_dataset = Subset(single_task, range(train_size, len(single_task)))

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    model = MaskTransformer(hidden_size=64,
                            segment=12,
                            task=None,
                            multi_task=True,
                            mask_ratio=1/3,
                            mask_strategy=1,
                            lr=3e-4,
                            weight_decay=1e-5,
                            n_layer=6,
                            n_head=8,
                            n_inner=64*4,
                            activation_function='relu',
                            n_positions=512,
                            n_ctx=512,
                            resid_pdrop=0.1,
                            attn_pdrop=0.1)
    
    trainer = pl.Trainer(max_epochs=10, strategy=DDPStrategy(find_unused_parameters=True), logger=True)
    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == '__main__':
    main()
