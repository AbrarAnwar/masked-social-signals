import torch
import torch.nn as nn

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