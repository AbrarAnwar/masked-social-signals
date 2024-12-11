from lightning import LightningModule
from models.autoencoder import AutoEncoder
from models.vqvae import VQVAE
from models.transformer import MaskTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import numpy as np


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


class Base_Module(LightningModule):
    def __init__(self, 
                segment,
                normalizer,
                lr,
                weight_decay):
        super(Base_Module, self).__init__()
        self.save_hyperparameters()
        self.segment = segment
        self.normalizer = normalizer
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=5,
                                                                            T_mult=2, 
                                                                            eta_min=1e-6)
        return [optimizer], [scheduler]


class AutoEncoder_Module(Base_Module):
    def __init__(self, task, 
                    segment, 
                    segment_length,
                    hidden_sizes, 
                    alpha, 
                    normalizer,
                    lr, 
                    weight_decay):
        super(AutoEncoder_Module, self).__init__(segment, normalizer, lr, weight_decay)
        self.save_hyperparameters()
        self.task = task
        self.segment_length = segment_length   
        self.model = AutoEncoder(FEATURES[self.task] * self.segment_length, hidden_sizes)

        self.alpha = alpha
    
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

        

class VQVAE_Module(Base_Module):
    def __init__(self, 
                hidden_sizes,
                h_dim,
                kernel,
                stride,
                res_h_dim,
                n_res_layers,
                n_embeddings,
                embedding_dim,
                temperature,
                beta,
                lamb,
                vq_weight,
                task,
                segment,
                segment_length,
                normalizer,
                lr,
                weight_decay,):
        super(VQVAE_Module, self).__init__(segment, normalizer, lr, weight_decay)
        self.save_hyperparameters()
        
        self.task = task
        self.segment_length = segment_length // 2 if task == 'pose' else segment_length
        

        self.model = VQVAE(hidden_sizes=hidden_sizes,
                            in_dim=FEATURES[self.task],
                            h_dim=h_dim,
                            kernel=kernel,
                            stride=stride,
                            res_h_dim=res_h_dim,
                            n_res_layers=n_res_layers,
                            n_embeddings=n_embeddings,
                            embedding_dim=embedding_dim,
                            segment_length=self.segment_length,
                            temperature=temperature,
                            beta=beta,
                            lamb=lamb,)
                            # frozen=True,
                            # pretrained='./pretrained2/main/30/pose/vqvae.pth')

        self.vq_weight = vq_weight
    
    def forward(self, batch):
        task = self.normalizer.minmax_normalize(batch[self.task], self.task)
        
        bz = task.size(0)
        # OG: task_reshaped = task.view(bz, 3, self.segment, self.segment_length, -1).view(bz*3*self.segment, self.segment_length, -1)
        #x = task_reshaped.view(bz*3*self.segment, self.segment_length, -1, 2). permute(0, 3, 1, 2)

        # do sampling on this
        if self.training:
            task_reshaped = task.view(bz, 3, self.segment, self.segment_length, -1)[:, :, ::3, :, :].view(bz*3*self.segment//3, self.segment_length, -1)
        else:
            task_reshaped = task.view(bz, 3, self.segment, self.segment_length, -1).view(bz*3*self.segment, self.segment_length, -1)

        vq_loss, x_hat, perplexity = self.model(task_reshaped) # 

        if self.training:
            x_hat = x_hat.view(bz, 3, self.segment*self.segment_length//3, -1)

        else:
            x_hat = x_hat.view(bz, 3, self.segment*self.segment_length, -1)
        
        return task, x_hat, vq_loss, perplexity

    def training_step(self, batch, batch_idx):
        y, y_hat, vq_loss, perplexity = self(batch)
        # velocity loss
        # y_vel = y[:, :, 1:, :] - y[:, :, :-1, :]
        # y_hat_vel = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]
        # velocity_loss = F.mse_loss(y_vel, y_hat_vel, reduction='mean')
        loss = F.mse_loss(y_hat, y, reduction='mean') + self.vq_weight * vq_loss # + velocity_loss
        # loss = vq_loss
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        self.log('perplexity', perplexity, on_epoch=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        y, y_hat, _, _ = self(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        return loss


class MaskTransformer_Module(Base_Module):
    def __init__(
            self,
            hidden_size,
            segment,
            segment_length,
            frozen,
            pretrained,
            feature_filling,
            normalizer,
            lr,
            weight_decay,
            feature_mask=[1,1,1,1,1,1],
            **kwargs
    ):
        super(MaskTransformer_Module, self).__init__(segment, normalizer, lr, weight_decay)
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.frozen = frozen

        self.feature_mask = torch.tensor(feature_mask) if type(feature_mask) == list else torch.tensor(FEATURE_MASK_DICT[feature_mask])

        self.model = MaskTransformer(hidden_size=hidden_size,
                                    segment=segment,
                                    segment_length=segment_length,
                                    frozen=frozen,
                                    pretrained=pretrained,
                                    feature_filling=feature_filling,
                                    feature_mask=self.feature_mask,
                                    **kwargs)

        self.task_weights = {task: 1.0 for task in self.model.task_list \
                             if task != 'word' and self.feature_mask[self.model.task_list.index(task)]}

        self.loss_weights = {'speaker': [0.1, 0.3], 
                        'bite': [0.1, 0.8]} # {'speaker': [0.1124, 0.314], 'bite': [0.0924, 0.793]}
    
        
    def forward(self, batch):
        for task in self.model.task_list[:-3]:
            batch[task] = self.normalizer.minmax_normalize(batch[task], task)
        return self.model(batch)

    def calculate_loss(self, y, y_hat, task_idx, task):
        if task == 'word':
            return None
        
        if task in ['gaze', 'headpose', 'pose']:
            current_y, current_y_hat = y[task_idx], y_hat[task_idx]

            # reconstruction loss
            task_loss = F.mse_loss(current_y_hat, current_y, reduction='mean')

            # velocity loss
            # y_vel = current_y[:, :, 1:, :] - current_y[:, :, :-1, :]
            # y_hat_vel = current_y_hat[:, :, 1:, :] - current_y_hat[:, :, :-1, :]
            # velocity = F.mse_loss(y_hat_vel, y_vel, reduction='mean')

            # # segment loss 
            # segment_length = current_y.size(2) // self.segment

            # y_segment_delta = current_y[:, :, segment_length-1:-1:segment_length, :] - current_y[:, :, segment_length::segment_length, :]
            # y_hat_segment_delta = current_y_hat[:, :, segment_length-1:-1:segment_length, :] - current_y_hat[:, :, segment_length::segment_length, :]
            # segment_loss = F.mse_loss(y_hat_segment_delta, y_segment_delta, reduction='mean')

            # task_loss = reconstruct_loss + velocity 
            # return task_loss

        elif task in ['speaker', 'bite']:
            weights = self.loss_weights[task]
            weights_matrix = torch.where(y[task_idx] == 0, weights[0], weights[1])
            task_loss = F.binary_cross_entropy_with_logits(y_hat[task_idx], y[task_idx], weight=weights_matrix)
        
        return task_loss
        # if self.training:
        #     return task_loss, self.calculate_grad_norm(task_loss)
        # return task_loss, 0
        
    # def calculate_grad_norm(self, task_loss):
    #     task_loss.backward(retain_graph=True)  

    #     grad_norm = torch.norm(
    #         torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.transformer.parameters() if p.grad is not None]), 2
    #     )
    #     return grad_norm
    
    # def adjust_task_weights(self, grad_norms):
    #     target_grad_norm = torch.tensor([1.0] * len(grad_norms))  

    #     relative_grad_norms = {task: grad_norms[task] / target_grad_norm[i]
    #                         for i, task in enumerate(grad_norms)}

    #     task_weight_updates = {task: self.task_weights[task] * relative_grad_norms[task] for task in grad_norms}

    #     total_weight = sum(task_weight_updates.values())
    #     self.task_weights = {task: weight / total_weight for task, weight in task_weight_updates.items()}


    def step(self, batch, loss_name):
        y, y_hat = self(batch)
        losses = dict()
        #grad_norms = dict()

        for task_idx, task in enumerate(self.model.task_list):
            if self.feature_mask[task_idx]:
                task_loss = self.calculate_loss(y, y_hat, task_idx, task)
                if task_loss:
                    self.log(f'{loss_name}/{task}', task_loss, on_epoch=True, sync_dist=True)
                    losses[task] = task_loss
        #             grad_norms[task] = grad_norm

        # if self.training:
        #     self.adjust_task_weights(grad_norms)
        #     # print(self.task_weights)
        #     total_loss = torch.stack([self.task_weights[task] * losses[task] for task in losses]).mean()

        # else:
        #     total_loss = torch.stack([losses[task] for task in losses]).mean()
        total_loss = torch.stack([losses[task] for task in losses]).mean()
        self.log(loss_name, total_loss, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val_loss')



class KMEANS_Module(Base_Module):
    def __init__(self, 
                hidden_sizes,
                task,
                frozen, 
                n_clusters,
                segment,
                segment_length,
                normalizer
                ):
        super(KMEANS_Module, self).__init__(segment_length, normalizer, 0, 0)
        self.save_hyperparameters()
        self.task = task
        self.segment = segment
        self.segment_length = segment_length // 2 if task == 'pose' else segment_length
        
        self.normalizer = normalizer
        self.frozen = frozen

        self.vqvae = VQVAE(hidden_sizes=hidden_sizes,
                           in_dim=26,
                           h_dim=128, 
                           kernel=3,
                           stride=1,
                           res_h_dim=32, 
                           n_res_layers=2,
                           n_embeddings=1024,
                           embedding_dim=1024,
                           segment_length=self.segment_length,
                           temperature=1,
                           beta=0.25,
                           lamb=0.25,
                           frozen=frozen,
                           pretrained='./pretrained2/main/30/pose/vqvae.pth')
        
        # 1152 = 32 * 3 * 12
        self.n_clusters = n_clusters
        self.coarse_kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.fine_kmeans = [MiniBatchKMeans(n_clusters=n_clusters) for _ in range(n_clusters)]

    
    @staticmethod
    def kmeans_perplexity(kmeans, data):
        """
        Calculate the perplexity of a fitted k-means model.

        Args:
            kmeans: A fitted sklearn MiniBatchKMeans or KMeans model.
            data: The data array of shape (num_samples, num_features).

        Returns:
            perplexity: The perplexity of the k-means model.
        """
        # Step 1: Get the distances to centroids
        distances = kmeans.transform(data)  # Shape: (num_samples, num_clusters)

        # Step 2: Assign each point to the closest centroid
        closest_clusters = np.argmin(distances, axis=1)

        # Step 3: Compute the squared distances for each point to its assigned centroid
        squared_distances = np.sum((data - kmeans.cluster_centers_[closest_clusters])**2, axis=1)

        # Step 4: Compute the log-likelihood
        log_likelihood = -0.5 * np.sum(squared_distances)

        # Step 5: Calculate perplexity
        num_samples = data.shape[0]
        perplexity = 2 ** (-log_likelihood / num_samples)

        return perplexity



    def forward(self, batch):
        task = self.normalizer.minmax_normalize(batch[self.task], self.task)
        
        bz = task.size(0)
        task_reshaped = task.view(bz, 3, self.segment, self.segment_length, -1).view(bz*3*self.segment, self.segment_length, -1)

        _, encoded, _ = self.vqvae.encode(task_reshaped)

        encoded = encoded.detach().cpu().numpy()

        coarse_perplexity, fine_perplexity = None, None

        if self.training:
            self.coarse_kmeans.partial_fit(encoded)

            coarse_labels = self.coarse_kmeans.predict(encoded)

            fine_perplexity_list = []
            for i in range(self.n_clusters):  # Iterate over coarse clusters
                cluster_data = encoded[coarse_labels == i]  # Data belonging to coarse cluster i
                if len(cluster_data) > 0:
                    n_clusters = min(len(cluster_data), self.fine_kmeans[i].n_clusters)
                    self.fine_kmeans[i].n_clusters = n_clusters
                    self.fine_kmeans[i].partial_fit(cluster_data)

                    fine_perplexity = self.kmeans_perplexity(self.fine_kmeans[i], cluster_data)
                    fine_perplexity_list.append(fine_perplexity)

            coarse_perplexity = self.kmeans_perplexity(self.coarse_kmeans, encoded)
            fine_perplexity = np.mean(fine_perplexity_list)

            # return task, encoded, coarse_perplexity, fine_perplexity
        
        coarse_labels = self.coarse_kmeans.predict(encoded)  # Shape: (batch_size,)

        reconstructed_embedding = []

        for idx, point in enumerate(encoded):
            coarse_id = coarse_labels[idx]
            if hasattr(self.fine_kmeans[coarse_id], 'cluster_centers_'):
                fine_id = self.fine_kmeans[coarse_id].predict(point.reshape(1,-1))[0]
                embedding = self.fine_kmeans[coarse_id].cluster_centers_[fine_id]
            else:
                embedding = self.coarse_kmeans.cluster_centers_[coarse_id]
            reconstructed_embedding.append(embedding)
        
        kmeans_encoded = torch.tensor(reconstructed_embedding).to(self.device)

        decoded = self.vqvae.decode(kmeans_encoded).view(bz, 3, self.segment*self.segment_length, -1)

        return task, decoded, coarse_perplexity, fine_perplexity
        

    def step(self, batch, loss_name):
        y, decoded, coarse_perplexity, fine_perplexity = self(batch)

        loss = F.mse_loss(decoded, y, reduction='mean')

        self.log(loss_name, loss, on_epoch=True, sync_dist=True)

        if self.training:
            self.log('coarse_perplexity', coarse_perplexity, on_epoch=True, sync_dist=True)
            self.log('fine_perplexity', fine_perplexity, on_epoch=True, sync_dist=True)
        
        if self.frozen:
            loss.requires_grad = True
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train_loss')


    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val_loss')