import utils
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import numpy as np


class MultiDataset(Dataset):
    def __init__(self, data_path, video_number, training=True):
        self.path = Path(data_path)
        self.files = []

        assert video_number != 9, 'Video 09 is empty'

        if training:
            for video_id in range(1,31):
                if video_id == 9 or video_id == video_number:
                    continue 

                video_folder = self.path / f"{video_id:02d}"
                for file in video_folder.glob('*.npz'):
                    self.files.append(file)

        else:
            video_folder = self.path / f"{video_number:02d}"
            for file in video_folder.glob('*.npz'):
                self.files.append(file)


    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        file = self.files[idx]
        data_loaded = np.load(file)
        word, status_speaker, whisper_speaker, headpose, gaze, pose, bite = [], [], [], [], [], [], []
        
        for i in range(3):
            word.append(torch.tensor(data_loaded[f'person_{i}_word'], dtype=torch.float32))
            status_speaker.append(torch.tensor(data_loaded[f'person_{i}_status_speaker'], dtype=torch.float32))
            whisper_speaker.append(torch.tensor(data_loaded[f'person_{i}_whisper_speaker'], dtype=torch.float32))
            headpose.append(torch.tensor(data_loaded[f'person_{i}_headpose'], dtype=torch.float32))
            gaze.append(torch.tensor(data_loaded[f'person_{i}_gaze'], dtype=torch.float32))
            pose.append(torch.tensor(data_loaded[f'person_{i}_pose'], dtype=torch.float32))
            bite.append(torch.tensor(data_loaded[f'person_{i}_bite'], dtype=torch.float32))
        
        
        return {'word': torch.stack(word), 
                'speaker': torch.stack(status_speaker), 
                'whisper_speaker': torch.stack(whisper_speaker), 
                'headpose': torch.stack(headpose), 
                'gaze': torch.stack(gaze), 
                'pose': torch.stack(pose), 
                'bite': torch.stack(bite)}

            
def get_loaders(batch_path, test_idx, batch_size=32, num_workers=2):
    dataset = MultiDataset(batch_path, test_idx, training=True)

    val_size = int(0.1 * len(dataset))
    start_idx = np.random.randint(0, len(dataset) - val_size)
    
    val_indices = list(range(start_idx, start_idx + val_size))
    train_indices = list(set(range(len(dataset))) - set(val_indices))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = MultiDataset(batch_path, test_idx, training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# testing

if __name__ == '__main__':
    batch_path = './dining_dataset/batch_window36_stride18_v4'
    test_idx = 29
    batch_size = 8
    num_workers = 2

    train_loader, val_loader, test_loader = get_loaders(batch_path, test_idx, batch_size, num_workers)

    # for batch_idx, batch in enumerate(val_loader):
    #     print('batch word:', batch['word'].shape)
    #     break
        
        





    