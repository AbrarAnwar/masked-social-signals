import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class MultiDataset(Dataset):
    def __init__(self, data_path, task=''):
        self.path = Path(data_path)
        self.person = 3
        self.task = task

    def __len__(self):
        return len(list(self.path.glob('*')))

    def __getitem__(self, idx):
        #print('idx', idx)
        file_path = self.path / f'{idx}.npz'
        data_loaded = np.load(file_path)
        segment = []
        
        for i in range(self.person):  
            person_data = {
                'word': data_loaded[f'person_{i}_word'].tolist(),
                'status_speaker': torch.tensor(data_loaded[f'person_{i}_status_speaker'], dtype=torch.float32),
                'whisper_speaker': torch.tensor(data_loaded[f'person_{i}_whisper_speaker'], dtype=torch.float32),
                'headpose': torch.tensor(data_loaded[f'person_{i}_headpose'], dtype=torch.float32),
                'gaze': torch.tensor(data_loaded[f'person_{i}_gaze'], dtype=torch.float32),
                'pose': torch.tensor(data_loaded[f'person_{i}_pose'], dtype=torch.float32),
            }
            # segment[i] = person_data

            # single task indexing
            if self.task:
                person_data = person_data[self.task]
            segment.append(person_data)
        
        if self.task and self.task != 'word':
            return torch.stack(segment)
        return segment
    
# list of dicts
# dicts of lists


    