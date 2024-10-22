import torch
import torch.nn as nn
from utils.utils import freeze

class LinearNet(nn.Module):
    def __init__(self, hidden_sizes, activation=True, frozen=False):
        super(LinearNet, self).__init__()

        layers = []
        prev_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
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


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class AutoEncoder(BaseModel):
    def __init__(self, hidden_sizes, 
                    pretrained=None,
                    frozen=False):
        super(AutoEncoder, self).__init__()
        self.encoder = LinearNet(hidden_sizes, activation=True)
        self.decoder = LinearNet(list(reversed(hidden_sizes)), activation=False)

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
        freeze(self.decoder)


