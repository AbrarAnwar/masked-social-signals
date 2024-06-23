import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
import lightning.pytorch as pl
import transformers

from models.autoencoder import Encoder, Decoder, AutoEncoder
from models.gpt2 import GPT2Model
from models.vqvae import VQVAE


class MaskTransformer(nn.Module):
    def __init__(
            self,
            normalizer,
            hidden_size,
            segment,
            frozen,
            pretrained,
            feature_filling,
            feature_mask=[1,1,1,1,1,1],
            **kwargs
    ):
        super().__init__()
        self.task_list = ['gaze', 'headpose', 'pose', 'word', 'speaker', 'bite']
        self.normalizer = normalizer
        self.segment = segment
        self.segment_length = 1080 // segment
        self.hidden_size = hidden_size
        self.feature_filling = feature_filling
        self.feature_mask = feature_mask
        self.padding_list = []

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        self.gaze_autoencoder = AutoEncoder(2 * 90, [128], frozen=frozen,
                                            pretrained='/home/tangyimi/masked-social-signals/pretrained_best/gaze/autoencoder.pth')
        self.headpose_autoencoder = AutoEncoder(2 * 90, [128], frozen=frozen,
                                            pretrained='/home/tangyimi/masked-social-signals/pretrained_best/headpose/autoencoder.pth')
        self.word_encoder = Encoder(768, [hidden_size], activation=False, frozen=frozen)

        self.pose_vqvae = VQVAE(h_dim=128, 
                           res_h_dim=32, 
                           n_res_layers=2,
                           n_embeddings=1024,
                           embedding_dim=64,
                           beta=0.25,
                           frozen=frozen,
                           pretrained='/home/tangyimi/masked-social-signals/pretrained_best/pose/vqvae32.pth')

        self.pose_projector = AutoEncoder(64*23*7, [4096, hidden_size], frozen=frozen)
        
        self.speaker_embedding = nn.Embedding(2, hidden_size)
        self.bite_embedding = nn.Embedding(2, hidden_size)
        
        self.speaker_classifier = nn.Linear(hidden_size, 1)
        self.bite_classifier = nn.Linear(hidden_size, 1)

        self.segment_embedding = nn.Embedding(segment, hidden_size)

    def encode(self, x, task):
        if task == 'pose':
            # CNN preprocess
            segment_length = self.segment_length // 2
            task_reshaped = x.view(self.bz, 3, self.segment, segment_length, -1)
            task_reshaped = task_reshaped.view(self.bz*3*self.segment, segment_length, -1)
            task_z = task_reshaped.view(self.bz*3*self.segment, segment_length, -1, 2). permute(0, 3, 1, 2)

            _, x_hat = self.pose_vqvae.encode(task_z)
            z = x_hat.permute(0, 2, 3, 1).reshape(self.bz*3*self.segment, -1)

            # projection
            x_hat = self.pose_projector.encode(z).view(self.bz*3, self.segment, self.hidden_size)

            return x, x_hat

        elif task == 'speaker':
            # embed speaker
            # if 30% of the segment is speaking, then the person is speaking 
            speaker_reshaped = x.reshape(self.bz, 3, self.segment, -1) # (self.bz, 3, 6, 180)
            speaker_sum = speaker_reshaped.sum(dim=-1) # (self.bz, 3, 180)
            speaker_tranformed = (speaker_sum > 0.3 * self.segment_length).float().unsqueeze(-1).reshape(-1, 1) 
            speaker_embed = self.speaker_embedding(speaker_tranformed.int()).reshape(self.bz*3, self.segment, -1) # (self.bz, 3, 6, hidden_size)

            return speaker_tranformed.view(self.bz, 3, self.segment, -1), speaker_embed

        elif task == 'bite':
            # embed bite
            # if the person is biting, then the value is 1
            bite_reshaped = x.reshape(self.bz, 3, self.segment, -1) # (self.bz*3, 6, 180)
            bite_sum = bite_reshaped.sum(dim=-1) # (self.bz, 3, 180)
            bite_tranformed = (bite_sum >= 1).float().unsqueeze(-1).reshape(-1, 1) 
            bite_embed = self.bite_embedding(bite_tranformed.int()).reshape(self.bz*3, self.segment, -1) # (self.bz, 3, 6, hidden_size)

            return bite_tranformed.view(self.bz, 3, self.segment, -1), bite_embed
        else:
            # same reshape for all other tasks
            # but different autoencoder for each task and returns
            task_z = x.reshape(self.bz, 3, self.segment, self.segment_length, -1) # (self.bz, 3, 6, 180, feature_dim)
            task_reshaped = task_z.reshape(self.bz*3*self.segment, -1) # (self.bz*3*6, 180, 2)

            if task == 'gaze':
                encode = self.gaze_autoencoder.encode(task_reshaped).view(self.bz*3, self.segment, -1) 
                return x, encode
            elif task == 'headpose':
                encode = self.headpose_autoencoder.encode(task_reshaped).view(self.bz*3, self.segment, -1) 
                return x, encode
            elif task == 'word':
                word_mean = torch.mean(task_z, dim=3)
                encode = self.word_encoder(word_mean).view(self.bz*3, self.segment, -1)
                return None, encode

    def decode(self, output, task):
        task_idx = self.task_list.index(task)
        task_output = output[:, task_idx::len(self.task_list), :]

        if task in self.padding_list:
            task_output = task_output[:, :, :128]
        
        task_output_reshaped = task_output.view(self.bz*3*self.segment, -1)

        if task == 'gaze':
            return self.gaze_autoencoder.decode(task_output_reshaped).view(self.bz, 3, self.segment*self.segment_length, -1)

        elif task == 'headpose':
            return self.headpose_autoencoder.decode(task_output_reshaped).view(self.bz, 3, self.segment*self.segment_length, -1)

        elif task == 'pose':
            task_output_reshaped = self.pose_projector.decode(task_output_reshaped).view(self.bz*3*self.segment, 23, 7, 64).permute(0, 3, 1, 2)
            return self.pose_vqvae.decode(task_output_reshaped).reshape(self.bz, 3, self.segment*self.segment_length // 2, -1)

        elif task == 'speaker':
            return self.speaker_classifier(task_output_reshaped).view(self.bz, 3, self.segment, -1)
        elif task == 'bite':
            return self.bite_classifier(task_output_reshaped).view(self.bz, 3, self.segment, -1)

    def padding(self, encode, task):

        if encode.size(-1) != self.hidden_size:
            if self.feature_filling == 'pad':
                encode_padded = F.pad(encode, (0, self.hidden_size - self.small_hidden_size))
            elif self.feature_filling == 'repeat':
                factor = self.hidden_size // encode.size(-1)
                encode_padded = torch.cat([encode]*factor, dim=2)
            else:
                raise NotImplementedError('Feature filling should be either pad or repeat')

            self.padding_list.append(task)
            return encode_padded

        return encode


    def forward(self, batch):
        for task in self.task_list[:-3]:
            batch[task] = self.normalizer.minmax_normalize(batch[task], task)

        self.bz = batch['gaze'].size(0)

        encode_list = []
        ys = []

        # data augmentation
        if self.training:
            random_sequences = torch.tensor([[0,1,2], [1,2,0], [2,0,1]])
            shuffled_people = random_sequences[torch.randint(0,3,(1,))].squeeze(0)
            batch = {k: v[:, shuffled_people, ...] for k, v in batch.items()}
        
        # encode all the tasks
        for task in self.task_list:
            current = batch[task]
            y, encode = self.encode(current, task)

            encode_padded = self.padding(encode, task)
            
            ys.append(y)
            encode_list.append(encode_padded)

        # it will make the input as (headpose1, gaze1, pose1, word1, headpose2, gaze2, pose2, word2, ...) (self.bz, 24, 64)
        stacked_inputs = torch.stack(encode_list, dim=1).permute(0, 2, 1, 3).reshape(self.bz*3, len(self.task_list)*self.segment, -1) # (self.bz*3, 24, 64)

        # add segment embeddings
        segment = torch.arange(self.segment).expand(self.bz*3, -1).to(stacked_inputs.device)
        segment_embeddings = self.segment_embedding(segment) 
        segment_embeddings_repeated = torch.repeat_interleave(segment_embeddings, len(self.task_list), dim=1) # (self.bz*3, 6*12, 64)
        
        stacked_inputs = stacked_inputs + segment_embeddings_repeated 

        # mask some of the inputs
        feature_mask = self.feature_mask.unsqueeze(0).unsqueeze(2).repeat(1, self.segment, self.hidden_size).expand(self.bz*3, -1, -1).to(stacked_inputs.device)
        stacked_inputs = stacked_inputs * feature_mask

        # it will make the input as (headpose_p1_t1, gaze_p1_t1, pose_p1_t1, word_p1_t1, headpose_p2_t1, gaze_p2_t1, pose_p2_t1, wordp2_t1, ...)
        stacked_inputs = stacked_inputs.view(self.bz, 3, -1, self.hidden_size).permute(0, 2, 1, 3).reshape(self.bz, -1, self.hidden_size) # (self.bz, 24*3, hidden_size)

        output = self.transformer(inputs_embeds=stacked_inputs)['last_hidden_state'] # (self.bz, 24*3, 64)

        output = output.view(self.bz, -1, 3, self.hidden_size).permute(0, 2, 1, 3).reshape(self.bz*3, -1, self.hidden_size) # (self.bz*3, 24, 64)

        y_hats = []
        # decode all the tasks
        for task in self.task_list:
            y_hat = self.decode(output, task)
            y_hats.append(y_hat)
        
        return ys, y_hats


