import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import freeze
from lightning import LightningModule


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation=True, frozen=False):
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

        if frozen:
            self.freeze()

    def forward(self, x):
        return self.network(x)

    def freeze(self):
        freeze(self.network)


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_sizes):
        super(Decoder, self).__init__()

        layers = []
        prev_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:] + [output_dim]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.pop()
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, 
                    hidden_sizes, 
                    pretrained=None,
                    frozen=False):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_sizes)
        self.decoder = Decoder(input_dim, hidden_sizes[::-1])

        if pretrained:
            self.load(pretrained)

        if frozen:
            self.freeze()
            

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def freeze(self):
        freeze(self.encoder)
        #freeze(self.decoder)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


