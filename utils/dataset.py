import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import numpy as np

class MultiDataset(Dataset):
    def __init__(self, data_path):
        self.path = Path(data_path)
        self.person = 3

    def __len__(self):
        return len(list(self.path.glob('*')))

    def __getitem__(self, idx):
        file_path = self.path / f'{idx}.npz'
        data_loaded = np.load(file_path)
        segment = dict()
        word, status_speaker, whisper_speaker, headpose, gaze, pose = [], [], [], [], [], []
        
        for i in range(self.person):
            word.append(data_loaded[f'person_{i}_word'].tolist())
            status_speaker.append(torch.tensor(data_loaded[f'person_{i}_status_speaker'], dtype=torch.float32))
            whisper_speaker.append(torch.tensor(data_loaded[f'person_{i}_whisper_speaker'], dtype=torch.float32))
            headpose.append(torch.tensor(data_loaded[f'person_{i}_headpose'], dtype=torch.float32))
            gaze.append(torch.tensor(data_loaded[f'person_{i}_gaze'], dtype=torch.float32))
            pose.append(torch.tensor(data_loaded[f'person_{i}_pose'], dtype=torch.float32))
        
        segment['word'] = word
        segment['status_speaker'] = torch.stack(status_speaker)
        segment['whisper_speaker'] = torch.stack(whisper_speaker)
        segment['headpose'] = torch.stack(headpose)
        segment['gaze'] = torch.stack(gaze)
        segment['pose'] = torch.stack(pose) 
        
        return segment


def custom_collate_fn(batch):
    word = [item['word'] for item in batch]
    batch = default_collate(batch)
    batch['word'] = word 
    return batch

    