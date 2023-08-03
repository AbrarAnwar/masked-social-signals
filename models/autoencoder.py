import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy

from utils.dataset import *
from torch.utils.data import DataLoader, Subset
from utils.normalize import *
from utils.visualize import *

from lightning import LightningModule

# 1-layer mlp hidden size 2340 -> 1024
# 2-layer mlp 2340 -> 1024 -> 512 (768)
# n-layer variable
# gesture 640 -> 200
# 768

'''
class Encoder(nn.Module):
    def __init__(self, feature_dim, segment_length, reduced_dim):
        super().__init__()
        self.linear = nn.Sequential(
                nn.Linear(segment_length*feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, reduced_dim)
            )
        
    def forward(self, batch):
        return self.linear(batch)
    

class Decoder(nn.Module):
    def __init__(self, hidden_size, segment_length, feature_dim):
        super().__init__()
        self.linear = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, segment_length*feature_dim)
            )
        
    def forward(self, batch):
        return self.linear(batch)
'''

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

    def __init__(self, task, segment, hidden_sizes, lr, weight_decay, batch_size):
        super().__init__()
        self.task = task
        self.batch_length = 1080
        self.segment = segment
        self.segment_length = int(self.batch_length / self.segment)
        self.hidden_size = copy.deepcopy(hidden_sizes)
        self.feature_dim = self.FEATURES[self.task]
        self.input_dim = self.segment_length * self.feature_dim
        self.encoder = Encoder(self.FEATURES[self.task] * self.segment_length, self.hidden_size)
        self.decoder = Decoder(self.segment_length * self.FEATURES[self.task], self.hidden_size[::-1])
    
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.batch_size = batch_size
        self.dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18')
        train_size = int(0.8 * len(self.dataset))
        self.train_dataset = Subset(self.dataset, range(0, train_size))
        self.test_dataset = Subset(self.dataset, range(train_size, len(self.dataset)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def on_train_start(self):
        self.normalizer = Normalizer(self.train_dataloader())

    def forward(self, batch):
        batch[self.task] = self.normalizer.minmax_normalize(batch[self.task], self.task)

        batch = batch[self.task]
        bz = batch.size(0)
        # take 15fps
        #batch = batch[:, :, ::2, :] # (bz, 3, 540, feature_dim)
        batch = batch.reshape(bz, 3, self.segment, self.segment_length, batch.size(-1)) # (bz, 3, segment, segment_length, feature_dim)

        x = batch.reshape(bz*3*self.segment, -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        decoded = decoded.reshape(bz, 3, self.segment, self.segment_length, -1)
        return batch, decoded

    def training_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        loss = F.mse_loss(y, y_hat, reduction='mean')
        return loss
    
    def on_test_start(self):
        self.test_losses = []

        result_dir = f'./result/autoencoder/{self.task}/hidden{self.hidden_size}_lr{self.lr}_wd{self.weight_decay}'
        model_dir = f'./pretrained/{self.task}/hidden{self.hidden_size}_lr{self.lr}_wd{self.weight_decay}'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.result_dir = result_dir
        self.model_dir = model_dir
        self.total_batches = len(self.test_dataloader())
        
    def test_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.test_losses.append(loss)
        
        if batch_idx >= self.total_batches - 2:
            output = 0
            # take the last segment
            y = y[:, :, -1, :, :].squeeze(2)
            y_hat = y_hat[:, :, -1, :, :].squeeze(2) # (bz, 3, segment_length, feature_dim)

            # denormalize
            y = self.normalizer.minmax_denormalize(y, self.task)
            y_hat = self.normalizer.minmax_denormalize(y_hat, self.task)

            videos_prediction = construct_batch_video(y_hat, task=self.task)
            videos_inference = construct_batch_video(y, task=self.task, color=(0,0,255))

            result_videos = np.concatenate([videos_prediction, videos_inference], axis=2)

            for i in range(batch[self.task].shape[0]):
                write_video(result_videos[i], f'{self.result_dir}/{batch_idx}_{output}.mp4', fps=30)
                output += 1

        return loss
    
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        self.log(f'lr{self.lr}_hidden{self.hidden_size}_wd{self.weight_decay}_{self.task}_test_loss', avg_loss) 
        self.test_losses.clear()
        # write the loss to json file
        with open(f'{self.model_dir}/{avg_loss.item()}.txt', 'w') as f:
            f.write(str(avg_loss.item()))

        torch.save(self.encoder.state_dict(), f"{self.model_dir}/encoder.pth")
        torch.save(self.decoder.state_dict(), f"{self.model_dir}/decoder.pth")


def main():
    
    for task in ['headpose', 'gaze']:
        model = AutoEncoder(task=task, segment=12, hidden_sizes=[128, 64], lr=3e-4, weight_decay=1e-5, batch_size=16)
        trainer = pl.Trainer(max_epochs=10, strategy=DDPStrategy(find_unused_parameters=True), logger=True)
        trainer.fit(model)
        trainer.test(model)
    '''
    for hidden_size in [64, 128, 80]:
        for lr in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
            for weight_decay in [1e-3, 1e-4, 1e-5]:
                model = AutoEncoder(task='pose', segment=12, hidden_size=hidden_size, lr=lr, weight_decay=weight_decay, batch_size=16)
                trainer = pl.Trainer(max_epochs=10, strategy=DDPStrategy(find_unused_parameters=True), logger=True)
                trainer.fit(model)
                trainer.test(model)
    
    for hidden_sizes in [[512], [512, 256], [768, 512], [768, 256], [768, 512, 256]]:
        for lr in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
            for weight_decay in [1e-3, 1e-4, 1e-5]:
                print(f'\n Testing {hidden_sizes}\n')
                model = AutoEncoder(task='pose', segment=24, hidden_sizes=hidden_sizes, lr=lr, weight_decay=weight_decay, batch_size=16)
                trainer = pl.Trainer(max_epochs=10, strategy=DDPStrategy(find_unused_parameters=True), logger=True)
                trainer.fit(model)
                trainer.test(model)
    
    for hidden_sizes in [[768, 256], [512]]:
        model = AutoEncoder(task='pose', segment=12, hidden_sizes=hidden_sizes, lr=3e-4, weight_decay=1e-5, batch_size=16)
        trainer = pl.Trainer(max_epochs=10, strategy=DDPStrategy(find_unused_parameters=True), logger=True)
        trainer.fit(model)
        trainer.test(model)
    '''
        
if __name__ == '__main__':
    main()


