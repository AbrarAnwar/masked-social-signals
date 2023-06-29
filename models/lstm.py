from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lightning import LightningModule
from lightning.pytorch.strategies import DDPStrategy
import lightning.pytorch as pl

from utils.dataset import *
from utils.visualize import *
from utils.embeddings import *
from models.base import Encoder, Decoder


class LSTMModel(LightningModule):
    FEATURES = {'word':768, 'headpose':2, 'gaze':2, 'pose':26}

    def __init__(self, reduced_dim, hidden_size, segment, task, multi_task, method, lr, weight_decay):
        super().__init__()
        self.task = task
        self.feature_dim = self.FEATURES[task]
        self.reduced_dim = reduced_dim
        self.hidden_size = hidden_size
        self.segment = segment
        self.batch_length = 1080
        self.segment_length = int(self.batch_length / self.segment)
        self.multi_task = multi_task
        self.method = method # 'concat' or 'maxpool'

        # testing
        self.output = 0
        self.test_losses = []

        if not self.multi_task:
            self.encoder = Encoder(self.feature_dim, self.segment_length, self.reduced_dim)
            self.lstm = nn.LSTM(input_size=self.reduced_dim, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
            self.decoder = Decoder(self.hidden_size*2, self.segment_length, self.feature_dim)
        else:
            self.processor = BERTProcessor()
            self.task_list = ['headpose', 'gaze', 'pose', 'word']
            self.encoder = nn.ModuleList([Encoder(self.FEATURES['headpose'], self.segment_length, self.reduced_dim),
                                        Encoder(self.FEATURES['gaze'], self.segment_length, self.reduced_dim),
                                        Encoder(self.FEATURES['pose'], self.segment_length, self.reduced_dim),
                                        Encoder(self.FEATURES['word'], self.segment_length, self.reduced_dim)])
            if self.method == 'concat':
                self.lstm = nn.LSTM(input_size=self.reduced_dim*4, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
            elif self.method == 'maxpool':
                self.lstm = nn.LSTM(input_size=self.reduced_dim, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
            
            self.decoder = Decoder(self.hidden_size*2, self.segment_length, self.feature_dim)

        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay

    
    def forward_single_task(self, batch):
        batch = batch[self.task]
        bz = batch.size(0)
        batch = batch.reshape(bz, 3, self.segment, self.segment_length, batch.size(-1))
        train_segment = self.segment - 1

        x = batch[:, :, :train_segment, :, :]
        y = batch[:, :, -1, :, :].squeeze(2)

        x = x.permute(0, 2, 1, 3, 4).reshape(bz*train_segment, -1)

        encode = self.encoder(x).view(bz, train_segment, self.reduced_dim)
        lstm_out, _ = self.lstm(encode)
        lstm_out = lstm_out[:, -1, :] # 
        reconstructed = self.decoder(lstm_out).reshape(bz, 3, self.segment_length, -1)

        return y, reconstructed
                        
    def forward_multi_task(self, batch):
        batch['word'] = self.processor.get_embeddings(batch['word']) 
        encode_list = []
        bz = batch['pose'].size(0)
        train_segment = self.segment - 1

        for task_idx, task in enumerate(self.task_list):
            current = batch[task]
            current = current.reshape(bz, 3, self.segment, self.segment_length, current.size(-1))
            x = current[:, :, :train_segment, :, :] # (bz, 3, 5, 180, feature_dim)
            x = x.permute(0, 2, 1, 3, 4).reshape(bz*train_segment, -1)

            encode = self.encoder[task_idx](x).view(bz, train_segment, self.reduced_dim)
            
            encode_list.append(encode)

            if task == self.task:
                y = current[:, :, -1, :, :].squeeze(2) # (bz, 3, 180, feature_dim)
        
        if self.method == 'concat':
            encode = torch.cat(encode_list, dim=2)

        elif self.method == 'maxpool':
            stacked = torch.stack(encode_list, dim=3)
            encode, _ = torch.max(stacked, dim=3)

        lstm_out, _ = self.lstm(encode)
        lstm_out = lstm_out[:, -1, :]
        reconstructed = self.decoder(lstm_out).reshape(bz, 3, self.segment_length, -1) # (bz, 3, 180, feature_dim)

        return y, reconstructed

        
    def forward(self, batch):
        if not self.multi_task:
            return self.forward_single_task(batch)
        return self.forward_multi_task(batch)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean') 
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        if self.multi_task:
            result_dir = f'./lstm_result/{self.method}/{self.task}'
        else:
            result_dir = f'./lstm_result/single/{self.task}'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        y, y_hat = self.forward(batch)
        videos_prediction = construct_batch_video(y_hat, task=self.task)
        videos_inference = construct_batch_video(y, task=self.task, color=(0,0,255))

        result_videos = np.concatenate([videos_prediction, videos_inference], axis=2)

        for i in range(batch[self.task].shape[0]):
            write_video(result_videos[i], f'{result_dir}/{self.output}.mp4')
            self.output += 1

        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.test_losses.append(loss)
        return loss
    
    # calculate the total test reconstruction loss
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        if self.multi_task:
            loss_name = f'lstm_{self.task}_{self.method}_test_loss'
        else:
            loss_name = f'lstm_{self.task}_single_test_loss'
        self.log(loss_name, avg_loss) 
        self.test_losses.clear()



