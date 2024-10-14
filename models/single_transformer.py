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
import wandb


from models.gpt2 import GPT2Model



# (Return_1, state_1, action_1, Return_2, state_2, ...)
class SingleTransformer(LightningModule):
    FEATURES = {'pose':26, 'headpose':2, 'gaze':2}

    def __init__(
            self,
            hidden_size,
            task,
            segment,
            eval_type,
            pretrained,
            frozen,
            lr,
            weight_decay,
            warmup_ratio,
            batch_size,
            alpha,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.hidden_size = hidden_size # pose:512, headpose&gaze: 128
        self.segment = segment
        self.batch_length = 540 if task == 'pose' else 1080
        self.segment_length = int(self.batch_length / self.segment)
        self.eval_type = eval_type 
        self.pretrained = pretrained
        self.frozen = frozen

        self.embed_timestep = nn.Embedding(self.segment, self.hidden_size)
        self.encoder = Encoder(self.FEATURES[self.task] * self.segment_length, [self.hidden_size])
        self.decoder = Decoder(self.FEATURES[self.task] * self.segment_length, [self.hidden_size])

        if self.pretrained:
            self.encoder.load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained_best/{self.task}/encoder.pth"))
            self.decoder.load_state_dict(torch.load(f"/home/tangyimi/masked_mine/pretrained_best/{self.task}/decoder.pth"))
            if self.frozen:
                freeze(self.encoder)
                freeze(self.decoder)
        

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
        train_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v2', 30, training=True)

        split = int(0.8 * len(train_dataset))
        train_indices = list(range(split))
        val_indices = list(range(split, len(train_dataset)))

        self.train_dataset = Subset(train_dataset, train_indices)
        self.val_dataset = Subset(train_dataset, val_indices)
        self.test_dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18_v2', 30, training=False)
        
        self.normalizer = Normalizer(self.train_dataloader())
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16, collate_fn=custom_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, shuffle=True, num_workers=16, collate_fn=custom_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, shuffle=True, num_workers=16, collate_fn=custom_collate_fn)
    
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
            
    def on_train_start(self):
        self.train_losses = []
        self.testing = False
        

    def forward(self, batch):
        batch[self.task] = self.normalizer.minmax_normalize(batch[self.task], self.task)
        if self.task == 'pose':
            batch['pose'] = smoothing(batch['pose'], 1080).to(self.device)
        current = batch[self.task]
        bz = current.size(0)

        time = torch.arange(self.segment).expand(bz*3, -1).to(self.device)
        time_embeddings = self.embed_timestep(time) # (bz*3, 6, 64)

        current = current.reshape(bz, 3, self.segment, self.segment_length, current.size(-1)) # (bz, 3, 12, 45, 26)

        current_reshaped = current.reshape(bz*3*self.segment, -1)

        encode = self.encoder(current_reshaped).view(bz*3, self.segment, -1) # (bz*3, 6, 64)
    
        encode_projection = encode + time_embeddings

        # (bz*3*12, 64) (bz, 3, 12, 64) (bz, 12,3,64) => pose1_p1, pose1_p2, pose_p3
        inputs = encode_projection.view(bz, 3, self.segment, self.hidden_size).permute(0, 2, 1, 3).reshape(bz, -1, self.hidden_size) # (bz, 12*3, hidden_size)
        
        outputs = self.transformer(inputs_embeds=inputs)['last_hidden_state'] # (bz, 24*3, 64)

        outputs = outputs.view(bz, -1, 3, self.hidden_size).permute(0, 2, 1, 3).reshape(bz*3, -1, self.hidden_size) # (bz*3, 12, 64)

        if self.testing:
            # take the last timestep to do visualize only
            if self.eval_type == 1:
                y = current.clone()[:, :, -1, :, :].squeeze(2) # (bz, 3, 12, 90, feature_dim)
                prediction = outputs[:, -1, :].squeeze(1) # (bz*3, 64)
                reconstructed = self.decoder(prediction).reshape(bz, 3, self.segment_length, -1) # (bz, 3, 45, 1170)
            elif self.eval_type == 2:
                random_index = torch.randint(0, self.segment, (1,)).item()
                y = current.clone()[:, :, random_index, :, :].squeeze(2) # (bz, 3, 90, feature_dim)
                prediction = outputs[:, random_index, :].squeeze(1)
                reconstructed = self.decoder(prediction).reshape(bz, 3, self.segment_length, -1)
            else:
                raise NotImplementedError('Mask strategy should be 1 or 2')
        else:
            # the same training process
            y = current.clone().view(bz, 3, self.segment*self.segment_length, -1) # (bz, 3, 12*45, 26)
            output_reshaped = outputs.reshape(bz*3*self.segment, -1)
            reconstructed = self.decoder(output_reshaped).view(bz, 3, self.segment*self.segment_length, -1) # (bz, 3, 12*45, 26)

        return y, reconstructed 
        
    
    def training_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch) # -1
        reconstruction_loss = F.mse_loss(y_hat, y, reduction='mean')
        # calculate velocity 
        y_vel = y[:, :, 1:, :] - y[:, :, :-1, :]
        y_hat_vel = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]
        velocity = F.mse_loss(y_vel, y_hat_vel, reduction='mean')
        loss = reconstruction_loss + self.alpha * velocity

        self.train_losses.append(loss)
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss
    
    def on_training_epoch_end(self):
        self.log('train_loss', torch.stack(self.train_losses).mean(), on_epoch=True, sync_dist=True)
    
    def on_validation_start(self):
        self.val_losses = []

    def validation_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.val_losses.append(loss)
        return loss
    
    def on_validation_epoch_end(self):  
        self.log('val_loss', torch.stack(self.val_losses).mean(), on_epoch=True, sync_dist=True)
    
    def on_test_start(self):
        root_dir = 'result'
        self.test_losses = []

        result_dir = f'./{root_dir}/transformer/single/{self.task}/lr{self.lr}_wd{self.weight_decay}_frozen{self.frozen}_warmup{self.warmup_ratio}_alpha{self.alpha}'
        
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.result_dir = result_dir
        self.testing = True


    def test_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.test_losses.append(loss)

        if batch_idx >= self.trainer.num_test_batches[0] - 2:
            if self.trainer.global_rank == 0:
                file_name = f'{self.result_dir}/{batch_idx}'
                visualize(self.task, self.normalizer, y, y_hat, file_name)

    def on_test_epoch_end(self):
        fps = 15 if self.task == 'pose' else 30
        for filename in os.listdir(self.result_dir):
            if filename.endswith(".mp4"):
                self.logger.experiment.log({f'{self.task}_video': wandb.Video(os.path.join(self.result_dir, filename), fps=fps, format="mp4")})
  
        avg_loss = torch.stack(self.test_losses).mean()
        loss_name = f'transformer_single_{self.task}_test_loss'
        self.log(loss_name, avg_loss)
        self.test_losses.clear()
        self.testing = False


def main():
    model = SingleTransformer(hidden_size=128,
                            task='gaze',
                            segment=12,
                            eval_type=2,
                            pretrained=True,
                            frozen=True,
                            lr=3e-4,
                            weight_decay=1e-5,
                            warmup_ratio=0.1,
                            batch_size=16,
                            alpha=1,
                            n_layer=6,
                            n_head=8,
                            n_inner=128*4,
                            activation_function='relu',
                            n_positions=36,
                            n_ctx=36,
                            resid_pdrop=0.1,
                            attn_pdrop=0.1,
                            n_bundle=3)
    
    wandb_logger = WandbLogger(project="sample")
    
    trainer = pl.Trainer(max_epochs=1, strategy=DDPStrategy(find_unused_parameters=True), logger=wandb_logger, num_sanity_val_steps=0,)
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    main()

