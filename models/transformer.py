import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR
import os

from lightning import LightningModule
from lightning.pytorch.strategies import DDPStrategy
import lightning.pytorch as pl
import transformers
from lightning.pytorch.loggers import WandbLogger

from torchmetrics import Accuracy, Precision, Recall, F1Score

from models.autoencoder import *
from models.gpt2 import GPT2Model
from utils.dataset import *
from utils.visualize import *
from utils.embeddings import *
from utils.normalize import *
from utils.utils import *
import wandb

from lightning import seed_everything


class MaskTransformer(LightningModule):
    FEATURES = {'word':768, 'headpose':2, 'gaze':2, 'pose':26}
    FEATURE_MASK_DICT = {'multi': [1,1,1,1,1,1],
                        'mask_headpose': [0,1,1,1,1,1],
                        'mask_gaze': [1,0,1,1,1,1],
                        'mask_pose': [1,1,0,1,1,1],
                        'mask_word': [1,1,1,0,1,1],
                        'mask_speaker': [1,1,1,1,0,1],
                        'mask_bite': [1,1,1,1,1,0],
                        'headpose_only': [1,0,0,0,0,0],
                        'gaze_only': [0,1,0,0,0,0],
                        'pose_only': [0,0,1,0,0,0],
                        'speaker_only': [0,0,0,0,1,0],
                        'bite_only': [0,0,0,0,0,1]}

    def __init__(
            self,
            hidden_size,
            segment,
            frozen,
            pretrained,
            feature_filling,
            lr,
            weight_decay,
            warmup_ratio,
            batch_size,
            alpha,
            result_root_dir,
            feature_mask=[1,1,1,1,1,1],
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.small_hidden_size = 128 # for headpose and gaze, depends on pretrained autoencoders
        self.frozen = frozen
        self.segment = segment
        self.batch_length = 1080 # 270
        self.segment_length = int(self.batch_length / self.segment)
        self.pretrained = pretrained
        self.feature_filling = feature_filling

        self.processors = BERTProcessor()
        self.embed_timestep = nn.Embedding(self.segment, self.hidden_size)

        self.embed_speaker = nn.Embedding(2, self.hidden_size)
        self.embed_bite = nn.Embedding(2, self.hidden_size)

        self.task_list = ['headpose', 'gaze', 'pose', 'word', 'speaker', 'bite'] 
        self.feature_mask = torch.Tensor(feature_mask) if type(feature_mask) == list else torch.Tensor(self.FEATURE_MASK_DICT[feature_mask])
        self.feature_mask_exclude_word = torch.cat((self.feature_mask[:3], self.feature_mask[4:])) # exclude word

        self.classifiers = nn.ModuleList([nn.Linear(self.hidden_size, 1),
                                          nn.Linear(self.hidden_size, 1)])

        self.encoder = nn.ModuleList([Encoder(self.FEATURES['headpose'] * 90, [self.small_hidden_size]),
                                    Encoder(self.FEATURES['gaze'] * 90, [self.small_hidden_size]),
                                    Encoder(self.FEATURES['pose'] * 45, [self.hidden_size]), # fps 15
                                    Encoder(self.FEATURES['word'] * 90, [self.hidden_size])])
        
        self.decoder = nn.ModuleList([Decoder(self.FEATURES['headpose'] * 90, [self.small_hidden_size]),
                                        Decoder(self.FEATURES['gaze'] * 90, [self.small_hidden_size]),
                                        Decoder(self.FEATURES['pose'] * 45, [self.hidden_size])])

        if self.pretrained:
            for task_idx, task in enumerate(self.task_list[:-3]):
                self.encoder[task_idx].load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained_best/{task}/encoder.pth"))
                self.decoder[task_idx].load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained_best/{task}/decoder.pth"))
            
            if self.frozen:
                for encoder, decoder in zip(self.encoder, self.decoder):
                    freeze(encoder)
                    freeze(decoder)
                    

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio 
        self.alpha = alpha

        # load data
        self.batch_size = batch_size
        self.val_batch_size = 8

        train_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v3', 30, training=True)
        #train_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window9_stride5_v2', 30, training=True)

        split = int(0.8 * len(train_dataset))
        train_indices = list(range(split))
        val_indices = list(range(split, len(train_dataset)))

        self.train_dataset = Subset(train_dataset, train_indices)
        self.val_dataset = Subset(train_dataset, val_indices)
        self.test_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v3', 30, training=False)
        
        self.normalizer = Normalizer(self.train_dataloader())

        self.root_dir = result_root_dir
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_lambda = lambda epoch: self.lr_schedule(self.global_step)
        scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [scheduler]

    def lr_schedule(self, current_step):
        total_steps = self.trainer.num_training_batches * self.trainer.max_epochs
        warmup_steps = self.warmup_ratio * total_steps
        if current_step < warmup_steps:
            lr_mult = float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return self.lr * lr_mult


    def forward_multi_task(self, batch):
        batch['word'] = self.processors.get_embeddings(batch['word'])
        batch['pose'] = smoothing(batch['pose'], self.batch_length).to(self.device)

        bz = batch['gaze'].size(0)

        encode_list = []
        original = []

        if self.training:
            random_sequences = torch.tensor([[0,1,2], [1,2,0], [2,0,1]])
            shuffled_people = random_sequences[torch.randint(0,3,(1,))].squeeze(0)
            batch = {k: v[:, shuffled_people, ...] for k, v in batch.items()}
        
        for task_idx, task in enumerate(self.task_list[:-2]):
            current = batch[task]
            if task == 'pose':
                segment_length = self.segment_length // 2
            else:
                segment_length = self.segment_length

            current = current.reshape(bz, 3, self.segment, segment_length, current.size(-1)) # (bz, 3, 6, 180, feature_dim)

            original.append(current.clone()) 

            current_reshaped = current.reshape(bz*3*self.segment, -1) # (bz*3*6, 180, 2)
            
            encode = self.encoder[task_idx](current_reshaped).view(bz*3, self.segment, -1) # (bz*3, 6, hidden_size)

            if task in ['headpose', 'gaze']:
                if self.feature_filling == 'pad':
                    encode = F.pad(encode, (0, self.hidden_size - self.small_hidden_size))
                elif self.feature_filling == 'repeat':
                    factor = self.hidden_size // self.small_hidden_size
                    encode = torch.cat([encode]*factor, dim=2)
                else:
                    raise NotImplementedError('Feature filling should be either pad or repeat')

            encode_list.append(encode)

        speaker = batch['speaker'] # (bz, 3, 1080)
        speaker_reshaped = speaker.reshape(bz, 3, self.segment, -1) # (bz, 3, 6, 180)
        speaker_sum = speaker_reshaped.sum(dim=-1) # (bz, 3, 180)
        speaker_tranformed = (speaker_sum > 0.3 * self.segment_length).float().unsqueeze(-1).reshape(-1, 1) # if 30% of the segment is speaking, then the person is speaking
        original.append(speaker_tranformed)

        speaker_embed = self.embed_speaker(speaker_tranformed.int()).reshape(bz*3, self.segment, -1) # (bz, 3, 6, hidden_size)
        encode_list.append(speaker_embed)

        bite = batch['bite'] # (bz, 3, 1080)
        bite_reshaped = bite.reshape(bz, 3, self.segment, -1) # (bz*3, 6, 180)
        bite_sum = bite_reshaped.sum(dim=-1) # (bz, 3, 180)
        bite_tranformed = (bite_sum >= 1).float().unsqueeze(-1).reshape(-1, 1) # if the person is biting, then the value is 1
        original.append(bite_tranformed)

        bite_embed = self.embed_bite(bite_tranformed.int()).reshape(bz*3, self.segment, -1) # (bz, 3, 6, hidden_size)
        encode_list.append(bite_embed)
        
        # it will make the input as (headpose1, gaze1, pose1, word1, headpose2, gaze2, pose2, word2, ...) (bz, 24, 64)
        stacked_inputs = torch.stack(encode_list, dim=1).permute(0, 2, 1, 3).reshape(bz*3, len(self.task_list)*self.segment, -1) # (bz*3, 24, 64)

        # add time embeddings
        time = torch.arange(self.segment).expand(bz*3, -1).to(self.device)
        time_embeddings = self.embed_timestep(time) 
        time_embeddings_repeated = torch.repeat_interleave(time_embeddings, len(self.task_list), dim=1) # (bz*3, 6*12, 64)
        
        stacked_inputs = stacked_inputs + time_embeddings_repeated 

        # mask some of the inputs
        feature_mask = self.feature_mask.unsqueeze(0).unsqueeze(2).repeat(1, self.segment, self.hidden_size).expand(bz*3, -1, -1).to(self.device)
        stacked_inputs = stacked_inputs * feature_mask

        # it will make the input as (headpose_p1_t1, gaze_p1_t1, pose_p1_t1, word_p1_t1, headpose_p2_t1, gaze_p2_t1, pose_p2_t1, wordp2_t1, ...)
        stacked_inputs = stacked_inputs.view(bz, 3, -1, self.hidden_size).permute(0, 2, 1, 3).reshape(bz, -1, self.hidden_size) # (bz, 24*3, hidden_size)

        output = self.transformer(inputs_embeds=stacked_inputs)['last_hidden_state'] # (bz, 24*3, 64)

        output = output.view(bz, -1, 3, self.hidden_size).permute(0, 2, 1, 3).reshape(bz*3, -1, self.hidden_size) # (bz*3, 24, 64)

        ys, y_hats = [], []
        for task_idx, task in enumerate(self.task_list[:-3]):
            
            task_output = output[:, task_idx::len(self.task_list), :]
            if task == 'pose':
                segment_length = self.segment_length // 2   
            else:
                task_output = task_output[:, :, :self.small_hidden_size]
                segment_length = self.segment_length

            task_output_reshaped = task_output.reshape(bz*3*self.segment, -1)

            decode = self.decoder[task_idx](task_output_reshaped).view(bz, 3, self.segment*segment_length, -1) # (bz, 3, 6*180, feature_dim)
            y = original[task_idx].view(bz, 3, self.segment*segment_length, -1)

            y_hats.append(decode)
            ys.append(y)

        # for binary classification tasks
        for task_idx, task in enumerate(self.task_list[-2:]):
            task_output = output[:, task_idx + 4::len(self.task_list), :]
            task_output_reshaped = task_output.reshape(bz*3*self.segment, -1)
            
            decode = self.classifiers[task_idx](task_output_reshaped).view(bz, 3, self.segment, -1)
            y = original[task_idx + 4].view(bz, 3, self.segment, -1)

            y_hats.append(decode)
            ys.append(y)
        
        return ys, y_hats
        

    def forward(self, batch):
        for task in self.task_list[:-3]:
            batch[task] = self.normalizer.minmax_normalize(batch[task], task)

        return self.forward_multi_task(batch)

    def _shared_step(self, batch, name):
        y, y_hat = self.forward(batch)
        losses = []

        for task_idx, task in enumerate(self.task_list[:-3]):
            if self.feature_mask_exclude_word[task_idx]:
                current_y, current_y_hat = y[task_idx], y_hat[task_idx]
                reconstruct_loss = F.mse_loss(current_y, current_y_hat, reduction='mean')

                y_vel = current_y[:, :, 1:, :] - current_y[:, :, :-1, :]
                y_hat_vel = current_y_hat[:, :, 1:, :] - current_y_hat[:, :, :-1, :]
                velocity = F.mse_loss(y_vel, y_hat_vel, reduction='mean')

                task_loss = reconstruct_loss + self.alpha * velocity
                
                self.log(f'{name}/{task}', task_loss, on_epoch=True, sync_dist=True)

                losses.append(task_loss)

        for task_idx, task in enumerate(self.task_list[-2:]):
            if self.feature_mask_exclude_word[task_idx - 2]:
                classification_loss = F.binary_cross_entropy_with_logits(y_hat[task_idx - 2], y[task_idx - 2])
                self.log(f'{name}/{task}', classification_loss, on_epoch=True, sync_dist=True)
                losses.append(classification_loss)

        loss = torch.stack(losses).mean()
        self.log(name, loss, on_epoch=True, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val_loss')

    def on_test_start(self):   
        result_dir = f'./{self.root_dir}/transformer/{self.feature_mask}/lr{self.lr}_wd{self.weight_decay}_frozen{self.frozen}_warmup{self.warmup_ratio}_feature{self.feature_filling}'
        
        for i in self.task_list[:-3]:
            if not os.path.exists(f'{result_dir}/{i}'):
                os.makedirs(f'{result_dir}/{i}')
        
        self.result_dir = result_dir

        self.classification_metrics = {'speaker': 
                                    {'accuracy': Accuracy(task="binary").to(self.device), 
                                    'precision': Precision(task="binary", average='weighted').to(self.device), 
                                    'recall': Recall(task="binary", average='weighted').to(self.device), 
                                    'f1': F1Score(task="binary", average='weighted').to(self.device)}, 
                                    'bite': 
                                    {'accuracy': Accuracy(task="binary").to(self.device), 
                                    'precision': Precision(task="binary", average='weighted').to(self.device), 
                                    'recall': Recall(task="binary", average='weighted').to(self.device), 
                                    'f1': F1Score(task="binary", average='weighted').to(self.device)}}

    def test_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        for task_idx, task in enumerate(self.task_list[:-3]):
            if self.feature_mask_exclude_word[task_idx]:
                current_y = y[task_idx]
                current_y_hat = y_hat[task_idx]
                current_loss = F.mse_loss(current_y_hat, current_y, reduction='mean')
                self.log(f'test_loss/{task}', current_loss, on_epoch=True, sync_dist=True)

                if batch_idx >= self.trainer.num_test_batches[0] - 2:
                    if self.trainer.global_rank == 0:
                        file_name = f'{self.result_dir}/{task}/{batch_idx}'
                        visualize(task, self.normalizer, current_y, current_y_hat, file_name)
        
        for task_idx, task in enumerate(self.task_list[-2:]):
            if self.feature_mask_exclude_word[task_idx - 2]:
                for metric in self.classification_metrics[task].values():
                    metric.update(y_hat[task_idx-2], y[task_idx-2])


    def on_test_epoch_end(self):
        for task_idx, task in enumerate(self.task_list[:-3]):
            if self.feature_mask_exclude_word[task_idx]:
                fps = 15 if task == 'pose' else 30
                for filename in os.listdir(f'{self.result_dir}/{task}'):
                    if filename.endswith(".mp4"):
                        self.logger.experiment.log({f'video/{task}': wandb.Video(os.path.join(f'{self.result_dir}/{task}', filename), fps=fps, format="mp4")})
        
        for task_idx, task in enumerate(self.task_list[-2:]):
            if self.feature_mask_exclude_word[task_idx - 2]:
                for metric_name, metric in self.classification_metrics[task].items():
                    self.log(f'test/{task}/{metric_name}', metric.compute(), on_epoch=True, sync_dist=True)

# testing
def main():
    seed_everything(42)
    model = MaskTransformer(hidden_size=512, 
                            segment=12, 
                            task=None, 
                            frozen=False, 
                            pretrained=True, 
                            feature_filling='repeat', 
                            lr=3e-4, 
                            weight_decay=1e-5, 
                            warmup_ratio=0.1, 
                            batch_size=4, 
                            alpha=1,
                            result_root_dir='sample_result2',
                            feature_mask='mask_word',
                            n_layer=1,
                            n_head=4,
                            n_inner=512*4,
                            activation_function='relu',
                            n_ctx=216,
                            resid_pdrop=0.1,
                            attn_pdrop=0.1,
                            n_bundle=18)
    
    wandb_logger = WandbLogger(project="sample")
    
    trainer = pl.Trainer(max_epochs=1, strategy=DDPStrategy(find_unused_parameters=True), logger=wandb_logger, num_sanity_val_steps=0,)
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    main()

