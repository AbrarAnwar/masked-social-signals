
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lightning import LightningModule
import lightning.pytorch as pl

from utils.dataset import *
from utils.visualize_headgaze import *



# TODO: Make LSTM model in here
class LSTMModel(LightningModule):

    def __init__(self, feature_dim, reduced_dim) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.reduced_dim = reduced_dim
        self.deconstruct = nn.Sequential(
            nn.Linear(5*3*180*self.feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 5*self.reduced_dim),
        )
        self.lstm = nn.LSTM(input_size=self.reduced_dim, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.reconstruct = nn.Sequential(
            nn.Linear(64*2, 256),
            nn.ReLU(),
            nn.Linear(256, 3*180*self.feature_dim),
        )

    def forward(self, batch):
        reduced = self.deconstruct(batch).view(batch.size(0), 5, self.reduced_dim)
        lstm_out, _ = self.lstm(reduced)
        lstm_out = lstm_out[:, -1, :] #
        reconstructed = self.reconstruct(lstm_out)

        return reconstructed

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def training_step(self, batch, batch_idx):
        # batch (16,3,1080,2)
        if batch.shape[-1] == 75:
            batch = index_pose(batch)

        bz = batch.size(0)
        batch = batch.reshape(bz, 3, 6, 180, batch.size(-1))

        # do segement
        x = batch[:, :, :5, :, :]
        y = batch[:, :, 5, :, :]

        # reshape
        x = x.permute(0, 2, 1, 3, 4).reshape(bz, 5, -1)
        x_flatten = x.view(x.size(0), -1)
        y = y.reshape(bz, -1)

        y_hat = self.forward(x_flatten)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.log('train_loss', loss)
        return loss

    #def validation_step(self, batch, batch_idx):
    #    pass

def main(feature_dim, reduced_dim, task):
    single_task = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18', task=task)
    # split train and val
    train_size = int(0.8 * len(single_task))

    train_dataset = Subset(single_task, range(0, train_size))
    val_dataset = Subset(single_task, range(train_size, len(single_task)))

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False) 

    # construct model
    model = LSTMModel(feature_dim, reduced_dim)
    
    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(model, train_dataloader)

    # evaluate
    count = 0
    for batch in val_dataloader:
        result = evaluate(model, batch, task)
        for i in range(batch.shape[0]):
            write_video(result[i], f'./result/{task}/{count}.mp4')
            count += 1
        break
    

if __name__ == '__main__':
    main(feature_dim=2, reduced_dim=128, task='gaze')
