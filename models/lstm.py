from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lightning import LightningModule
import lightning.pytorch as pl

from utils.dataset import *
from utils.visualize import *
from utils.embeddings import *

class Encoder(nn.Module):
    def __init__(self, feature_dim, segment_length, reduced_dim):
        super().__init__()
        self.linear = nn.Sequential(
                nn.Linear(3*segment_length*feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, reduced_dim)
            )
        
    def forward(self, batch):
        return self.linear(batch)
    

class Decoder(nn.Module):
    def __init__(self, hidden_size, segment_length, feature_dim):
        super().__init__()
        self.linear = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 3*segment_length*feature_dim)
            )
        
    def forward(self, batch):
        return self.linear(batch)



# TODO: Make LSTM model in here
class LSTMModel(LightningModule):
    FEATURES = {'word':768, 'headpose':2, 'gaze':2, 'pose':26}

    def __init__(self, reduced_dim, hidden_size, segment, task, multi_task, method):
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

        self.output = 0

        if not self.multi_task:
            self.encoder = Encoder(self.feature_dim, self.segment_length, self.reduced_dim)
            self.lstm = nn.LSTM(input_size=self.reduced_dim, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
            self.decoder = Decoder(self.hidden_size*2, self.segment_length, self.feature_dim)
        else:
            self.processor = BERTProcessor()
            self.encoder = nn.ModuleList([Encoder(self.FEATURES['headpose'], self.segment_length, self.reduced_dim),
                                        Encoder(self.FEATURES['gaze'], self.segment_length, self.reduced_dim),
                                        Encoder(self.FEATURES['pose'], self.segment_length, self.reduced_dim),
                                        Encoder(self.FEATURES['word'], self.segment_length, self.reduced_dim)])
            if self.method == 'concat':
                self.lstm = nn.LSTM(input_size=self.reduced_dim*4, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
            elif self.method == 'maxpool':
                self.lstm = nn.LSTM(input_size=self.reduced_dim, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
            
            self.decoder = Decoder(self.hidden_size*2, self.segment_length, self.feature_dim)

    
    def forward_single_task(self, batch):
        batch = batch[self.task]
        bz = batch.size(0)
        batch = batch.reshape(bz, 3, self.segment, self.segment_length, batch.size(-1))
        train_segment = self.segment - 1

        x = batch[:, :, :train_segment, :, :]
        y = batch[:, :, -1, :, :].squeeze()

        x = x.permute(0, 2, 1, 3, 4).reshape(bz*train_segment, -1)

        encode = self.encoder(x).view(bz, train_segment, self.reduced_dim)
        lstm_out, _ = self.lstm(encode)
        lstm_out = lstm_out[:, -1, :] 
        reconstructed = self.decoder(lstm_out).reshape(bz, 3, self.segment_length, -1)

        return y, reconstructed
                        
    def forward_multi_task(self, batch):
        batch['word'] = self.processor.get_embeddings(batch['word']) 
        task_list = ['headpose', 'gaze', 'pose', 'word']
        encode_list = []
        bz = batch['pose'].size(0)
        train_segment = self.segment - 1

        for task_idx, task in enumerate(task_list):
            current = batch[task]
            current = current.reshape(bz, 3, self.segment, self.segment_length, current.size(-1))
            x = current[:, :, :train_segment, :, :]
            x = x.permute(0, 2, 1, 3, 4).reshape(bz*train_segment, -1)

            encode = self.encoder[task_idx](x).view(bz, train_segment, self.reduced_dim)
            
            encode_list.append(encode)

            if task == self.task:
                y = current[:, :, -1, :, :].squeeze()
        
        if self.method == 'concat':
            encode = torch.cat(encode_list, dim=2)

        elif self.method == 'maxpool':
            stacked = torch.stack(encode_list, dim=3)
            encode, _ = torch.max(stacked, dim=3)

        lstm_out, _ = self.lstm(encode)
        lstm_out = lstm_out[:, -1, :]
        reconstructed = self.decoder(lstm_out).reshape(bz, 3, self.segment_length, -1)

        return y, reconstructed

        
    def forward(self, batch):
        if not self.multi_task:
            return self.forward_single_task(batch)
        return self.forward_multi_task(batch)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)

    def training_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean') 
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        videos_prediction = construct_batch_video(y_hat, task=self.task)
        videos_inference = construct_batch_video(y, task=self.task, color=(0,0,255))

        result_videos = np.concatenate([videos_prediction, videos_inference], axis=2)

        for i in range(batch[self.task].shape[0]):
            write_video(result_videos[i], f'./result_/{self.task}/{self.output}.mp4')
            self.output += 1


def main(epoch, reduced_dim, hidden_size, segment, task, multi_task, method):
    # load data
    single_task = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18') 
    
    train_size = int(0.8 * len(single_task))
    train_dataset = Subset(single_task, range(0, train_size))
    val_dataset = Subset(single_task, range(train_size, len(single_task)))

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn) 

    # construct model
    model = LSTMModel(reduced_dim, hidden_size, segment, task, multi_task=multi_task, method=method)
    
    trainer = pl.Trainer(max_epochs=epoch)
    trainer.fit(model, train_dataloader)
    trainer.test(model, val_dataloader)


if __name__ == '__main__':
    main(1, 64, 64, 6, 'pose', multi_task=True, method='concat')

