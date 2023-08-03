import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lightning import LightningModule

from utils.dataset import *
from utils.visualize import *
from utils.embeddings import *
from utils.normalize import *
from models.autoencoder import *


class LSTMModel(LightningModule):
    FEATURES = {'word':768, 'headpose':2, 'gaze':2, 'pose':26}

    def __init__(self, 
                 reduced_dim, 
                 hidden_size, 
                 segment, 
                 task, 
                 multi_task, 
                 method, 
                 pretrained, 
                 lr, 
                 weight_decay, 
                 warmup_steps,
                 batch_size):
        
        super().__init__()
        assert task is not None, 'task should be specified'
        self.task = task
        self.feature_dim = self.FEATURES[task]
        self.reduced_dim = reduced_dim
        self.hidden_size = hidden_size
        self.segment = segment
        self.batch_length = 1080
        self.segment_length = int(self.batch_length / self.segment)
        self.multi_task = multi_task
        self.method = method # 'concat' or 'maxpool'
        self.pretrained = pretrained

        if self.multi_task:
            self.processor = BERTProcessor()
            self.task_list = ['headpose', 'gaze', 'pose', 'word']
            self.encoder = nn.ModuleList([Encoder(self.FEATURES['headpose'], self.segment_length, self.reduced_dim),
                                        Encoder(self.FEATURES['gaze'], self.segment_length, self.reduced_dim),
                                        Encoder(self.FEATURES['pose'], self.segment_length, self.reduced_dim),
                                        Encoder(self.FEATURES['word'], self.segment_length, self.reduced_dim)])
            if self.method == 'concat':
                self.lstm = nn.LSTM(input_size=self.reduced_dim*4, hidden_size=self.hidden_size, num_layers=2, batch_first=True, )#bidirectional=True)
            elif self.method == 'maxpool':
                self.lstm = nn.LSTM(input_size=self.reduced_dim, hidden_size=self.hidden_size, num_layers=2, batch_first=True, )#bidirectional=True)
            else:
                raise ValueError('method should be concat or maxpool')
            
            self.decoder = Decoder(self.hidden_size, self.segment_length, self.feature_dim)

            if self.pretrained:
                for i, task in enumerate(self.task_list):
                    self.encoder[i].load_state_dict(torch.load(f"./pretrained/{task}/encoder.pth"))
                    self.encoder[i].requires_grad_(False)
                self.decoder.load_state_dict(torch.load(f"./pretrained/{self.task}/decoder.pth"))
                self.decoder.requires_grad_(False)
        else:       
            self.lstm = nn.LSTM(input_size=self.reduced_dim, hidden_size=self.hidden_size, num_layers=2, batch_first=True, )#bidirectional=True)
            
            self.encoder = Encoder(self.feature_dim, self.segment_length, self.reduced_dim)
            self.decoder = Decoder(self.hidden_size, self.segment_length, self.feature_dim)

            if self.pretrained:
                self.encoder.load_state_dict(torch.load(f"./pretrained/{self.task}/encoder.pth"))
                self.decoder.load_state_dict(torch.load(f"./pretrained/{self.task}/decoder.pth"))
                self.encoder.requires_grad_(False)
                self.decoder.requires_grad_(False)

        # optimizer
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
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    def configure_optimizers(self):
        betas = (0.9, 0.95)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=betas)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda steps: min((steps+1)/self.warmup_steps, 1))
        return optimizer

    def on_train_start(self):
        self.normalizer = Normalizer(self.train_dataloader())

    def forward_single_task(self, batch): 
        batch[self.task] = self.normalizer.minmax_normalize(batch[self.task], self.task)
        batch = batch[self.task]
        bz = batch.size(0)
        batch = batch.reshape(bz, 3, self.segment, self.segment_length, batch.size(-1))
        train_segment = self.segment - 1

        x = batch[:, :, :train_segment, :, :] # (bz, 3, 5, 180, feature_dim)
        y = batch[:, :, -1, :, :].squeeze(2)

        x = x.reshape(bz*3*train_segment, -1)

        encode = self.encoder(x).view(-1, train_segment, self.reduced_dim) # (bz*3, 5, reduced_dim)
        lstm_out, _ = self.lstm(encode)
        lstm_out = lstm_out[:, -1, :] 
        reconstructed = self.decoder(lstm_out).reshape(bz, 3, self.segment_length, -1)

        return y, reconstructed
                        
    def forward_multi_task(self, batch):
        batch['word'] = self.processor.get_embeddings(batch['word'])
        for task in self.task_list[:-1]:
            batch[task] = self.normalizer.minmax_normalize(batch[task], task)

        encode_list = []
        bz = batch['gaze'].size(0)
        train_segment = self.segment - 1

        for task_idx, task in enumerate(self.task_list):
            current = batch[task]
            current = current.reshape(bz, 3, self.segment, self.segment_length, current.size(-1))
            x = current[:, :, :train_segment, :, :] # (bz, 3, 5, 180, feature_dim)
            x = x.reshape(bz*3*train_segment, -1)

            encode = self.encoder[task_idx](x).view(-1, train_segment, self.reduced_dim) # (bz*3, 5, reduced_dim)
            
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
        if self.multi_task:
            return self.forward_multi_task(batch) 
        return self.forward_single_task(batch)

    def training_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean') 
        self.log('train_loss', loss)
        return loss
    
    # TODO: folder naming
    def on_test_start(self):
        root_dir = 'sample'
        if self.multi_task:
            if self.pretrained:
                result_dir = f'./{root_dir}/lstm/{self.method}_segment{self.segment}_pretrained/{self.task}'
            else:
                result_dir = f'./{root_dir}/lstm/{self.method}_segment{self.segment}/{self.task}'
        else:
            if self.pretrained:
                result_dir = f'./{root_dir}/lstm/single_segment{self.segment}_pretrained/{self.task}'
            else:
                result_dir = f'./{root_dir}/lstm/single_segment{self.segment}/{self.task}'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.result_dir = result_dir
        self.test_losses = []
        self.total_batches = len(self.test_dataloader())
    
    def test_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.test_losses.append(loss)

        if batch_idx >= self.total_batches - 3:
            output = 0
            # denormalize
            y = self.normalizer.minmax_denormalize(y, self.task)
            y_hat = self.normalizer.minmax_denormalize(y_hat, self.task)

            videos_prediction = construct_batch_video(y_hat, task=self.task)
            videos_inference = construct_batch_video(y, task=self.task, color=(0,0,255))

            result_videos = np.concatenate([videos_prediction, videos_inference], axis=2)

            for i in range(batch[self.task].shape[0]):
                write_video(result_videos[i], f'{self.result_dir}/{batch_idx}_{output}.mp4')
                output += 1

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



