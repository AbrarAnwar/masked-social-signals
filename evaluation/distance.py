import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from experiment.module import AutoEncoder_Module, VQVAE_Module, MaskTransformer_Module
from models.vqvae import VQVAE


def plot_distance_matrix(pretrained):
    model = VQVAE(hidden_sizes=[1024],
                           in_dim=26,
                           h_dim=128, 
                           kernel=3,
                           stride=1,
                           res_h_dim=32, 
                           n_res_layers=2,
                           n_embeddings=512,
                           embedding_dim=32,
                           segment_length=45,
                           beta=0.25,
                           frozen=True,
                           pretrained=pretrained)

    codebook = model.vector_quantization.embedding.weight

    cosine_sim = F.cosine_similarity(codebook.unsqueeze(1), codebook.unsqueeze(0), dim=-1)
    distance_matrix = 1 - cosine_sim

    plt.figure(figsize=(100, 100))  # Adjust the figure size based on the number of embeddings
    sns.heatmap(distance_matrix.numpy(), cmap="coolwarm", square=True, cbar=True, annot=False)
    plt.title('Pairwise Distance Between Embeddings')
    
    plt.savefig(f'distance_matrix.png', dpi=300)
    mask = ~torch.eye(distance_matrix.size(0), dtype=bool)
    
    # Extract the off-diagonal elements
    off_diagonal_values = distance_matrix[mask]
    
    # Compute min, max, and mean
    min_value = off_diagonal_values.min().item()
    max_value = off_diagonal_values.max().item()
    mean_value = off_diagonal_values.mean().item()
    std_value = off_diagonal_values.std().item()
    
    print(f'Minimum distance: {min_value}')
    print(f'Maximum distance: {max_value}')
    print(f'Mean distance: {mean_value}')
    print(f'Standard deviation: {std_value}')

if __name__ == '__main__':
    plot_distance_matrix('pretrained/main/30/pose/vqvae.pth')
