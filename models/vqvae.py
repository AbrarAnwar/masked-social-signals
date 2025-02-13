import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import freeze
from models.autoencoder import AutoEncoder, BaseModel


class CNNEncoder(nn.Module):

    def __init__(self, in_dim, h_dim, kernel, stride, n_res_layers, res_h_dim):
        super(CNNEncoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_dim, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)


class CNNDecoder(nn.Module):

    def __init__(self, in_dim, h_dim, kernel, stride, n_res_layers, res_h_dim):
        super(CNNDecoder, self).__init__()
        self.inverse_conv_stack = nn.Sequential(
            ResidualStack(in_dim,in_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose1d(
                in_dim, h_dim, kernel_size=kernel, stride=stride, padding=1), #output_padding=1),
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)

class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e # 512
        self.e_dim = e_dim # 64
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, hard=True):
        # reshape z -> (batch, height, width, channel) and flatten
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        # find closest encodings
        if hard:
            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_e).to(z.device)
            min_encodings.scatter_(1, min_encoding_indices, 1)

            # get quantized latent vectors
            z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        else:
            # soft selection
            softmax_d = F.softmax(-d, dim=1)
            z_q = torch.matmul(softmax_d, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        
        if hard:
            # preserve gradients
            z_q = z + (z_q - z).detach()

            # perplexity
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        else:
            perplexity = None

        return loss, z_q, perplexity # min_encodings, min_encoding_indices



class VQVAE(BaseModel):
    def __init__(self, 
                hidden_sizes,
                in_dim,
                h_dim, 
                kernel,
                stride,
                res_h_dim, 
                n_res_layers,
                n_embeddings, 
                embedding_dim, 
                segment_length,
                beta, 
                pretrained=None,
                frozen=False
                ):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = CNNEncoder(in_dim, h_dim, kernel, stride, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv1d(
            h_dim, 32, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)

        self.linear_projector = AutoEncoder([32 * segment_length] + hidden_sizes)
        # decode the discrete latent representation
        self.decoder = CNNDecoder(32, in_dim, kernel, stride, n_res_layers, res_h_dim)

        if pretrained:
            self.load(pretrained)

        if frozen:
            self.freeze()

    def forward(self, x):

        embedding_loss1, z_q, _ = self.encode(x)
        embedding_loss2, x_hat, perplexity = self.decode(z_q)

        return (embedding_loss1 + embedding_loss2) / 2 , x_hat, perplexity

    
    def encode(self, x):
        x = x.permute(0, 2, 1).contiguous()
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e) #.permute(0, 2, 1).contiguous() 

        # TODO: modify encode
        embedding_loss, z_q, perplexity = self.vector_quantization(z_e)
        self.hidden_shape = z_q.shape

        z_q = z_q.flatten(start_dim=1) # (1152=bz*3*12, 32*90) -> (1152, 1024) -> transformer -> (1152, 1024)
        linear_proj = self.linear_projector.encode(z_q)

        return embedding_loss, linear_proj, perplexity


    def decode(self, z, hard=True): # (bz, 3, 12, 1024)
        linear_proj = self.linear_projector.decode(z)
        z_reshaped = linear_proj.view(self.hidden_shape)
        embedding_loss, z_e, perplexity = self.vector_quantization(z_reshaped, hard=hard)
        # import pdb; pdb.set_trace()
        x_hat = self.decoder(z_e).permute(0, 2, 1).contiguous()
        return embedding_loss, x_hat, perplexity
        

    def freeze(self):
        freeze(self.encoder)
        freeze(self.pre_quantization_conv)
        freeze(self.vector_quantization.embedding)
        freeze(self.linear_projector.encoder)
        freeze(self.linear_projector.decoder)
        freeze(self.decoder)

