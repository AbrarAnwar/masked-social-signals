import wandb
import random

pose = batch['pose']
pose = pose[:, :, :45, :]

self.batch_size = batch_size
        #dataset = MultiDataset('/data/tangyimi/batch_window36_stride18') 
        dataset = MultiDataset('/home/tangyimi/social_signal/dining_dataset/batch_window36_stride18')

        self.train_dataset, self.val_dataset, self.test_dataset = \
                utils.random_split(dataset, [.8, .1, .1], generator=torch.Generator().manual_seed(42)) 
        
        self.normalizer = Normalizer(self.train_dataloader())

DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)