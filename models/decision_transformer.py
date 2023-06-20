import numpy as np
import torch
import torch.nn as nn

from lightning import LightningModule
import transformers

from lstm import *
from gpt2 import GPT2Model

# (Return_1, state_1, action_1, Return_2, state_2, ...)
class DecisionTransformer(LightningModule):

    FEATURES = {'word':768, 'headpose':2, 'gaze':2, 'pose':26}
    def __init__(
            self,
            reduced_dim,
            hidden_size,
            task,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.transformer = GPT2Model(config)
        self.task = task
        self.feature_dim = self.FEATURES[self.task]
        self.reduced_dim = reduced_dim
        self.hidden_size = hidden_size

        #self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.encoder = Encoder(self.feature_dim, self.reduced_dim)

        self.decoder = Decoder(self.hidden_size, self.feature_dim)

    def forward(self, batch, attention_mask=None):

        batch = batch[self.task]
        bz = batch.size(0)
        batch = batch.reshape(bz, 3, 6, 180, batch.size(-1))

        x = batch[:, :, :5, :, :]
        y = batch[:, :, 5, :, :].squeeze()

        x = x.permute(0, 2, 1, 3, 4).reshape(bz*5, -1)

        if attention_mask is None:
            attention_mask = torch.ones((bz, 5), dtype=torch.long)

        #time_embeddings = self.embed_timestep(timesteps)
        # time embeddings are treated similar to positional embeddings
        #stacked_inputs = torch.stack(
        #    (returns_embeddings, state_embeddings, action_embeddings), dim=1
        #).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        #stacked_inputs = self.embed_ln(stacked_inputs)
        #stacked_attention_mask = torch.stack(
        #    (attention_mask, attention_mask, attention_mask), dim=1
        #).permute(0, 2, 1).reshape(bz, 5)

        encode = self.encoder(x).view(bz, 5, self.reduced_dim)
        transformer_outputs = self.transformer(
            inputs_embeds=encode,
            attention_mask=attention_mask,
        )
        y_hat = transformer_outputs['last_hidden_state']

        reconstructed = self.decoder(y_hat).reshape(bz, 3, 180, -1)
        
        return y, reconstructed
    
    def training_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean') 
        self.log('train_loss', loss)
        return loss

