from lightning import LightningModule
from models.autoencoder import AutoEncoder
from models.vqvae import VQVAE
from models.transformer import MaskTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURES = {'headpose':2, 'gaze':2, 'pose':26}
FEATURE_MASK_DICT = {'multi': [1,1,1,1,1,1],
                    'mask_gaze': [0,1,1,1,1,1],
                    'mask_headpose': [1,0,1,1,1,1],
                    'mask_pose': [1,1,0,1,1,1],
                    'mask_word': [1,1,1,0,1,1],
                    'mask_speaker': [1,1,1,1,0,1],
                    'mask_bite': [1,1,1,1,1,0],
                    'gaze_only': [1,0,0,0,0,0],
                    'headpose_only': [0,1,0,0,0,0],
                    'pose_only': [0,0,1,0,0,0],
                    'speaker_only': [0,0,0,0,1,0],
                    'bite_only': [0,0,0,0,0,1]}

class AutoEncoder_Module(LightningModule):
    def __init__(self, task, 
                    segment, 
                    segment_length,
                    normalizer,
                    hidden_sizes, 
                    alpha, 
                    lr, 
                    weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.segment = segment
        self.segment_length = segment_length   
        self.autoencoder = AutoEncoder(FEATURES[self.task] * self.segment_length, hidden_sizes)

        self.normalizer = normalizer

        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=self.trainer.max_epochs) 
        return [optimizer], [scheduler]
        
    
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) 
    

    def forward(self, batch):
        batch[self.task] = self.normalizer.minmax_normalize(batch[self.task], self.task)

        batch = batch[self.task]
        bz = batch.size(0)

        y = batch.clone()
        batch = batch.reshape(bz, 3, self.segment, self.segment_length, batch.size(-1)) # (bz, 3, segment, segment_length, feature_dim)

        x = batch.reshape(bz*3*self.segment, -1)
        decoded = self.autoencoder(x)

        decoded = decoded.reshape(bz, 3, self.segment*self.segment_length, -1)
        return y, decoded

    def training_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        reconstruction_loss = F.mse_loss(y, y_hat, reduction='mean')
        
        y_vel = y[:, :, 1:, :] - y[:, :, :-1, :]
        y_hat_vel = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]
        velocity_loss = F.mse_loss(y_vel, y_hat_vel, reduction='mean')
        loss = reconstruction_loss + self.alpha * velocity_loss

        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        return loss

        

class VQVAE_Module(LightningModule):
    def __init__(self, 
                normalizer,
                h_dim,
                res_h_dim,
                n_res_layers,
                n_embeddings,
                embedding_dim,
                beta,
                lr,
                weight_decay,
                task,
                segment,
                segment_length,):
        super(VQVAE_Module, self).__init__()
        self.save_hyperparameters()
        self.normalizer = normalizer
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.task = task
        self.segment = segment
        self.segment_length = segment_length
        
        self.model = VQVAE(h_dim, 
                           res_h_dim, 
                           n_res_layers,
                           n_embeddings,
                           embedding_dim,
                           beta)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=5,
                                                                            T_mult=2, 
                                                                            eta_min=1e-6)
        return [optimizer], [scheduler]

    def forward(self, batch):
        task = self.normalizer.minmax_normalize(batch[self.task], self.task)
        
        bz = task.size(0)
        task_reshaped = task.view(bz, 3, self.segment, self.segment_length, -1)
        task_reshaped = task_reshaped.view(bz*3*self.segment, self.segment_length, -1)
        x = task_reshaped.view(bz*3*self.segment, self.segment_length, -1, 2). permute(0, 3, 1, 2)
        embedding_loss, x_hat = self.model(x)
        x_hat = x_hat.permute(0, 2, 3, 1).contiguous().view(bz, 3, self.segment*self.segment_length, -1)
        
        return task, x_hat, embedding_loss

    def training_step(self, batch, batch_idx):
        y, y_hat, embedding_loss = self(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean') + embedding_loss
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        y, y_hat, _ = self(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        return loss


class MaskTransformer_Module(LightningModule):

    def __init__(
            self,
            hidden_size,
            segment,
            frozen,
            pretrained,
            feature_filling,
            lr,
            weight_decay,
            alpha, 
            normalizer,
            feature_mask=[1,1,1,1,1,1],
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.frozen = frozen
        self.segment = segment

        self.feature_mask = torch.Tensor(feature_mask) if type(feature_mask) == list else torch.Tensor(FEATURE_MASK_DICT[feature_mask])

        self.model = MaskTransformer(normalizer=normalizer,
                                    hidden_size=hidden_size,
                                    segment=segment,
                                    frozen=frozen,
                                    pretrained=pretrained,
                                    feature_filling=feature_filling,
                                    feature_mask=self.feature_mask,
                                    **kwargs)

        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=5,
                                                                            T_mult=2, 
                                                                            eta_min=1e-6)
        return [optimizer], [scheduler]
    
        
    def forward(self, batch):
        return self.model(batch)

    def calculate_loss(self, y, y_hat, task, training):
        task_idx = self.model.task_list.index(task)
        if self.feature_mask[task_idx]:
            if task in ['gaze', 'headpose', 'pose']:
                current_y, current_y_hat = y[task_idx], y_hat[task_idx]

                # reconstruction loss
                reconstruct_loss = F.mse_loss(current_y_hat, current_y, reduction='mean')

                if not training:
                    return reconstruct_loss

                # velocity loss
                y_vel = current_y[:, :, 1:, :] - current_y[:, :, :-1, :]
                y_hat_vel = current_y_hat[:, :, 1:, :] - current_y_hat[:, :, :-1, :]
                velocity = F.mse_loss(y_hat_vel, y_vel, reduction='mean')

                # segment loss 
                segment_length = current_y.size(2) // self.segment

                y_segment_delta = current_y[:, :, segment_length-1:-1:segment_length, :] - current_y[:, :, segment_length::segment_length, :]
                y_hat_segment_delta = current_y_hat[:, :, segment_length-1:-1:segment_length, :] - current_y_hat[:, :, segment_length::segment_length, :]
                segment_loss = F.mse_loss(y_hat_segment_delta, y_segment_delta, reduction='mean')

                task_loss = reconstruct_loss + velocity + self.alpha * segment_loss
                return task_loss

            elif task in ['speaker', 'bite']:
                classification_loss = F.binary_cross_entropy_with_logits(y_hat[task_idx], y[task_idx])
                return classification_loss
        
    def step(self, batch, loss_name, training):
        y, y_hat = self(batch)
        losses = []

        for task in self.model.task_list:
            
            task_loss = self.calculate_loss(y, y_hat, task, training)
            if task_loss:
                self.log(f'{loss_name}/{task}', task_loss, on_epoch=True, sync_dist=True)
                losses.append(task_loss)

        loss = torch.stack(losses).mean()
        self.log(loss_name, loss, on_epoch=True, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train_loss', training=True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val_loss', training=False)

    # def on_test_start(self):
    #     result_dir = f"./{self.root_dir}/{self.feature_mask}/{self.experiment_name}"
        
    #     for i in self.task_list[:-3]:
    #         os.makedirs(f'{result_dir}/{i}', exist_ok=True)
        
    #     self.result_dir = result_dir

    #     self.visualize_idx = torch.randint(low=0, high=self.trainer.num_test_batches[0], size=(1,)).item()
    #     self.classification_metrics = {'speaker': 
    #                                 {'accuracy': Accuracy(task="binary").to(self.device), 
    #                                 'precision': Precision(task="binary", average='weighted').to(self.device), 
    #                                 'recall': Recall(task="binary", average='weighted').to(self.device), 
    #                                 'f1': F1Score(task="binary", average='weighted').to(self.device)}, 
    #                                 'bite': 
    #                                 {'accuracy': Accuracy(task="binary").to(self.device), 
    #                                 'precision': Precision(task="binary", average='weighted').to(self.device), 
    #                                 'recall': Recall(task="binary", average='weighted').to(self.device), 
    #                                 'f1': F1Score(task="binary", average='weighted').to(self.device)}}
        
        
    # def test_step(self, batch, batch_idx):
    #     y, y_hat = self.forward(batch)
    #     for task_idx, task in enumerate(self.task_list[:-3]):
    #         if self.feature_mask_exclude_word[task_idx]:
    #             current_y = y[task_idx]
    #             current_y_hat = y_hat[task_idx]
    #             current_loss = F.mse_loss(current_y_hat, current_y, reduction='mean')
                
    #             self.log(f'test_loss/{task}', current_loss, on_epoch=True, sync_dist=True)
                
    #             #visualize one batch
    #             if batch_idx == self.visualize_idx:
    #                 file_name = f'{self.result_dir}/{task}/{batch_idx}'
    #                 visualize(task, self.normalizer, current_y, current_y_hat, file_name)
        
    #     for task_idx, task in enumerate(self.task_list[-2:]):
    #         if self.feature_mask_exclude_word[task_idx - 2]:
    #             for metric in self.classification_metrics[task].values():
    #                 metric.update(y_hat[task_idx-2], y[task_idx-2])


    # def on_test_epoch_end(self):
    #     for task_idx, task in enumerate(self.task_list[:-3]):
    #         if self.feature_mask_exclude_word[task_idx]:
    #             fps = 15 if task == 'pose' else 30
    #             for filename in os.listdir(f'{self.result_dir}/{task}'):
    #                 self.logger.experiment.log({f'video/{task}': wandb.Video(os.path.join(f'{self.result_dir}/{task}', filename), 
    #                                                                             fps=fps, 
    #                                                                             format="mp4")})
    #     for task_idx, task in enumerate(self.task_list[-2:]):
    #         if self.feature_mask_exclude_word[task_idx - 2]:
    #             for metric_name, metric in self.classification_metrics[task].items():
    #                 self.log(f'test/{task}/{metric_name}', metric.compute(), sync_dist=True)

