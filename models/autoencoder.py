import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import freeze
#from models.vqvae import VectorQuantizer


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation=True, frozen=False):
        super(LinearEncoder, self).__init__()

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


class LinearDecoder(nn.Module):
    def __init__(self, output_dim, hidden_sizes):
        super(LinearDecoder, self).__init__()

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

# write a residual block using mlp
class RedisualBlock(nn.Module):
    def __init__(self, in_dim, h_dims):
        super(RedisualBlock, self).__init__()
        self.block = AutoEncoder(in_dim, h_dims)

    def forward(self, x):
        return x + self.block(x)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class AutoEncoder(BaseModel):
    def __init__(self, input_dim, 
                    hidden_sizes, 
                    pretrained=None,
                    frozen=False):
        super(AutoEncoder, self).__init__()
        self.encoder = LinearEncoder(input_dim, hidden_sizes)
        self.decoder = LinearDecoder(input_dim, hidden_sizes[::-1])

        if pretrained:
            self.load(pretrained)

        # if frozen:
        #     self.freeze()
        freeze(self.encoder)
        if frozen:
            freeze(self.decoder)
            

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

'''
class VQVAE(BaseModel):
    def __init__(self, input_dim, 
                    hidden_sizes, 
                    n_embeddings,
                    beta,
                    pretrained=None,
                    frozen=False):
        super(VQVAE, self).__init__()
        self.encoder = LinearEncoder(input_dim, hidden_sizes)
        self.encoder_residual_block = RedisualBlock(hidden_sizes[-1], [256])

        self.vector_quantization = VectorQuantizer(
            n_embeddings, hidden_sizes[-1], beta)

        self.decoder_residual_block = RedisualBlock(hidden_sizes[-1], [256])
        self.decoder = LinearDecoder(input_dim, hidden_sizes[::-1])
        

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.encoder_residual_block(z_e)

        z_e = z_e.unsqueeze(1).unsqueeze(1)
        embedding_loss, z_q = self.vector_quantization(z_e)
        z_q = z_q.squeeze(1).squeeze(1)

        x_hat = self.decoder_residual_block(x_hat)
        x_hat = self.decoder(z_q)
        

        return embedding_loss, x_hat

'''
