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

    def __init__(self, n_e, e_dim, temperature, beta, lamb):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e # 512
        self.e_dim = e_dim # 64
        self.temperature = temperature

        self.beta = beta
        self.lamb = lamb

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # kaiming 
        nn.init.kaiming_normal_(self.embedding.weight)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        # min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # min_encodings = torch.zeros(
        #     min_encoding_indices.shape[0], self.n_e).to(z.device)
        # min_encodings.scatter_(1, min_encoding_indices, 1)

        # # get quantized latent vectors
        # z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        gumbel_softmax_probs = F.gumbel_softmax(
            -d, tau=self.temperature, hard=True
        )

        # Get quantized latent vectors using Straight-Through Estimator
        z_q = torch.matmul(gumbel_softmax_probs, self.embedding.weight)

        # Reshape z_q to match z's original shape
        z_q = z_q.view(z.shape)
        # compute loss for embedding

        # Compute the Gram matrix of embeddings
        embedding_weights = self.embedding.weight  # (n_e, e_dim)
        gram_matrix = torch.matmul(embedding_weights, embedding_weights.T)  # (n_e, n_e)

        # Identity matrix
        identity = torch.eye(self.n_e, device=embedding_weights.device)  # (n_e, n_e)

        # Compute the orthogonality loss and normalize by n^2
        orthogonal_loss = torch.mean((gram_matrix - identity) ** 2) / (self.n_e ** 2)

        # loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        #     torch.mean((z_q - z.detach()) ** 2)

        embedding_loss = torch.mean((z_q.detach() - z) ** 2)
        commitment_loss = self.beta * torch.mean((z_q - z.detach()) ** 2)

        loss = embedding_loss + self.beta * commitment_loss + self.lamb * orthogonal_loss

        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(gumbel_softmax_probs, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
            

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
                temperature,
                beta, 
                lamb,
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
            n_embeddings, embedding_dim, temperature, beta, lamb)
        # self.vector_quantization_second = VectorQuantizer(
        #     n_embeddings, embedding_dim, temperature, beta, lamb)

        # self.linear_projector = AutoEncoder([32 * segment_length] + hidden_sizes)
        # decode the discrete latent representation
        self.decoder = CNNDecoder(32, in_dim, kernel, stride, n_res_layers, res_h_dim)

        if pretrained:
            self.load(pretrained)

        if frozen:
            self.freeze()

    def forward(self, x):

        vq_loss, z_q, perplexity = self.encode(x)
        x_hat = self.decode(z_q)

        return vq_loss, x_hat, perplexity

    
    def encode(self, x):
        x = x.permute(0, 2, 1).contiguous()
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e) #.permute(0, 2, 1).contiguous() (1152, 32, 45)
        self.hidden_shape = z_e.shape
        # import pdb; pdb.set_trace()

        # z_e_flatten = z_e.flatten(start_dim=1) # (1152, 576)
        # linear_proj = self.linear_projector.encode(z_e_flatten) # (1152, 1024)
        vq_loss1, z_q1, perplexity1 = self.vector_quantization(z_e)
        # vq_loss2, z_q2, perplexity2 = self.vector_quantization_second(linear_proj - z_q1.detach())

        # return vq_loss1 + vq_loss2, z_q1 + z_q2, perplexity1 + perplexity2
        return vq_loss1, z_q1, perplexity1

    def decode(self, z): # (bz, 3, 12, 1024)
        #embedding_loss, z_e, perplexity = self.vector_quantization(z, hard=hard)
        # z = self.linear_projector.decode(z)
        # z_e = z.view(self.hidden_shape)
        x_hat = self.decoder(z).permute(0, 2, 1).contiguous()
        return x_hat
        

    def freeze(self):
        freeze(self.encoder)
        freeze(self.pre_quantization_conv)
        freeze(self.vector_quantization.embedding)
        freeze(self.linear_projector.encoder)
        freeze(self.linear_projector.decoder)
        freeze(self.decoder)

