import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from models.autoencoder import LinearEncoder, LinearDecoder, AutoEncoder
from models.gpt2 import GPT2Model
from models.vqvae import VQVAE


class MaskTransformer(nn.Module):
    def __init__(
            self,
            hidden_size,
            segment,
            segment_length,
            frozen,
            pretrained,
            feature_filling,
            feature_mask=[1,1,1,1,1,1],
            **kwargs
    ):
        super().__init__()
        self.task_list = ['gaze', 'headpose', 'pose', 'word', 'speaker', 'bite']
        self.segment = segment
        self.segment_length = segment_length
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

        # self.gaze_autoencoder = AutoEncoder(2 * 90, [128], frozen=frozen,
        #                                     pretrained='/home/tangyimi/masked-social-signals/pretrained_best/gaze/autoencoder.pth')
        # self.headpose_autoencoder = AutoEncoder(2 * 90, [128], frozen=frozen,
        #                                     pretrained='/home/tangyimi/masked-social-signals/pretrained_best/headpose/autoencoder.pth')
        self.gaze_vqvae = VQVAE(hidden_sizes=[hidden_size],
                                    in_dim=2,
                                    h_dim=128,
                                    kernel=3,
                                    stride=1,
                                    res_h_dim=32,
                                    n_res_layers=2,
                                    n_embeddings=512,
                                    embedding_dim=32,
                                    segment_length=self.segment_length,
                                    beta=0.25,
                                    frozen=frozen,
                                    pretrained=f'./{pretrained}/gaze/vqvae.pth')
        self.gaze_projector = LinearEncoder(hidden_size, [768, hidden_size], activation=False)

        self.headpose_vqvae = VQVAE(hidden_sizes=[hidden_size],
                                    in_dim=2,
                                    h_dim=128,
                                    kernel=3,
                                    stride=1,
                                    res_h_dim=32,
                                    n_res_layers=2,
                                    n_embeddings=512,
                                    embedding_dim=32,
                                    segment_length=self.segment_length,
                                    beta=0.25,
                                    frozen=frozen,
                                    pretrained=f'./{pretrained}/headpose/vqvae.pth')
        self.headpose_projector = LinearEncoder(hidden_size, [768, hidden_size], activation=False)


        self.pose_vqvae = VQVAE(hidden_sizes=[hidden_size],
                           in_dim=26,
                           h_dim=128, 
                           kernel=3,
                           stride=1,
                           res_h_dim=32, 
                           n_res_layers=2,
                           n_embeddings=512,
                           embedding_dim=32,
                           segment_length=self.segment_length//2,
                           beta=0.25,
                           frozen=frozen,
                           pretrained=f'./{pretrained}/pose/vqvae.pth')
        self.pose_projector = LinearEncoder(hidden_size, [768, hidden_size], activation=False)


        self.word_encoder = LinearEncoder(768, [hidden_size], activation=False, frozen=frozen)
        self.speaker_embedding = nn.Embedding(2, hidden_size)
        self.bite_embedding = nn.Embedding(2, hidden_size)
        
        self.speaker_classifier = LinearEncoder(hidden_size, [hidden_size//2, 1], activation=False)
        self.bite_classifier = LinearEncoder(hidden_size, [hidden_size//2, 1], activation=False)

        self.segment_embedding = nn.Embedding(segment, hidden_size)

    def encode(self, x, task):
        if task == 'gaze':
            task_reshaped = x.view(self.bz, 3, self.segment, self.segment_length, -1).view(self.bz*3*self.segment, self.segment_length, -1)

            _, x_hat, _, min_encoding_indices = self.gaze_vqvae.encode(task_reshaped)

            return x, x_hat.view(self.bz*3, self.segment, -1)#, min_encoding_indices
        
        elif task == 'headpose':
            task_reshaped = x.view(self.bz, 3, self.segment, self.segment_length, -1).view(self.bz*3*self.segment, self.segment_length, -1)

            _, x_hat, _, min_encoding_indices = self.headpose_vqvae.encode(task_reshaped)

            return x, x_hat.view(self.bz*3, self.segment, -1)#, min_encoding_indices

        elif task == 'pose':
            segment_length = self.segment_length // 2
            task_reshaped = x.view(self.bz, 3, self.segment, segment_length, -1).view(self.bz*3*self.segment, segment_length, -1)

            _, x_hat, _, min_encoding_indices = self.pose_vqvae.encode(task_reshaped)

            return x, x_hat.view(self.bz*3, self.segment, -1)#, min_encoding_indices

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
        elif task == 'word':
            task_z = x.reshape(self.bz, 3, self.segment, self.segment_length, -1) # (self.bz, 3, 6, 180, feature_dim)
            task_reshaped = task_z.reshape(self.bz*3*self.segment, -1) # (self.bz*3*6, 180, 2)

            word_mean = torch.mean(task_z, dim=3)
            encode = self.word_encoder(word_mean).view(self.bz*3, self.segment, -1)
            return None, encode

    def decode(self, output, task):
        task_idx = self.task_list.index(task)
        task_output = output[:, task_idx::len(self.task_list), :]

        if task in self.padding_list:
            if self.feature_filling == 'pad':
                task_output = task_output[:, :, :128]
            elif self.feature_filling == 'repeat':
                task_output = task_output.reshape(task_output.size(0), task_output.size(1), -1, 128).mean(dim=2)
        #import pdb; pdb.set_trace()
        task_output_reshaped = task_output.contiguous().view(self.bz*3*self.segment, -1) # (bz*3*12, 1024)

        if task == 'gaze':
            task_output_reshaped = self.gaze_projector(task_output_reshaped) # (bz*3*12, 1024) -> (bz*3*12, 512)
            _, x_hat, _, dist = self.gaze_vqvae.decode(task_output_reshaped)
            return x_hat.view(self.bz, 3, self.segment*self.segment_length, -1), dist

        elif task == 'headpose':
            task_output_reshaped = self.headpose_projector(task_output_reshaped)
            _, x_hat, _, dist = self.headpose_vqvae.decode(task_output_reshaped)
            return x_hat.view(self.bz, 3, self.segment*self.segment_length, -1), dist

        elif task == 'pose':
            task_output_reshaped = self.pose_projector(task_output_reshaped)
            _, x_hat, _, dist = self.pose_vqvae.decode(task_output_reshaped)
            return x_hat.view(self.bz, 3, self.segment*self.segment_length//2, -1), dist

        elif task == 'speaker':
            return self.speaker_classifier(task_output_reshaped).view(self.bz, 3, self.segment, -1), None

        elif task == 'bite':
            return self.bite_classifier(task_output_reshaped).view(self.bz, 3, self.segment, -1), None

        return None, None

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

        # it will make the input as (gaze1, headpose1, pose1, word1, gaze2, headpose2, pose2, word2, ...) (self.bz, 24, 64)
        stacked_inputs = torch.stack(encode_list, dim=1).permute(0, 2, 1, 3).reshape(self.bz*3, len(self.task_list)*self.segment, -1) # (self.bz*3, 24, 64)

        # add segment embeddings
        segment = torch.arange(self.segment).expand(self.bz*3, -1).to(stacked_inputs.device)
        segment_embeddings = self.segment_embedding(segment) 
        segment_embeddings_repeated = torch.repeat_interleave(segment_embeddings, len(self.task_list), dim=1) # (self.bz*3, 6*12, 64)
        
        stacked_inputs = stacked_inputs + segment_embeddings_repeated 

        # mask some of the inputs
        feature_mask = self.feature_mask.unsqueeze(0).unsqueeze(2).repeat(1, self.segment, self.hidden_size).expand(self.bz*3, -1, -1).to(stacked_inputs.device)

        stacked_inputs = stacked_inputs * feature_mask

        # it will make the input as (gaze_p1_t1, headpose_p1_t1, pose_p1_t1, word_p1_t1, gaze_p2_t1, headpose_p2_t1, pose_p2_t1, wordp2_t1, ...)
        stacked_inputs = stacked_inputs.view(self.bz, 3, -1, self.hidden_size).permute(0, 2, 1, 3).reshape(self.bz, -1, self.hidden_size) # (self.bz, 24*3, hidden_size)

        output = self.transformer(inputs_embeds=stacked_inputs)['last_hidden_state'] # (self.bz, 24*3, 64)

        output = output.view(self.bz, -1, 3, self.hidden_size).permute(0, 2, 1, 3).reshape(self.bz*3, -1, self.hidden_size) # (self.bz*3, 24, 64)

        y_hats = []
        dist_loss = []
        # decode all the tasks
        for task in self.task_list:
            y_hat, dist = self.decode(output, task)
            y_hats.append(y_hat)
            dist_loss.append(dist)
        
        return ys, y_hats, dist_loss


