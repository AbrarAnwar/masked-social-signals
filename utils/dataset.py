import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import numpy as np


def custom_collate_fn(batch):
    word = [item['word'] for item in batch]
    batch = default_collate(batch)
    batch['word'] = word 
    return batch


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
                if video_folder.exists():
                    for file in video_folder.glob('*.npz'):
                        self.files.append(file)

        else:
            video_folder = self.path / f"{video_number:02d}"
            if video_folder.exists():
                for file in video_folder.glob('*.npz'):
                    self.files.append(file)


    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        file = self.files[idx]
        data_loaded = np.load(file)
        segment = dict()
        word, status_speaker, whisper_speaker, headpose, gaze, pose, bite = [], [], [], [], [], [], []
        
        for i in range(3):
            word.append(data_loaded[f'person_{i}_word'].tolist())
            status_speaker.append(torch.tensor(data_loaded[f'person_{i}_status_speaker'], dtype=torch.float32))
            whisper_speaker.append(torch.tensor(data_loaded[f'person_{i}_whisper_speaker'], dtype=torch.float32))
            headpose.append(torch.tensor(data_loaded[f'person_{i}_headpose'], dtype=torch.float32))
            gaze.append(torch.tensor(data_loaded[f'person_{i}_gaze'], dtype=torch.float32))
            pose.append(torch.tensor(data_loaded[f'person_{i}_pose'], dtype=torch.float32))
            bite.append(torch.tensor(data_loaded[f'person_{i}_bite'], dtype=torch.float32))
        
        segment['word'] = word
        segment['speaker'] = torch.stack(status_speaker)
        segment['whisper_speaker'] = torch.stack(whisper_speaker)
        segment['headpose'] = torch.stack(headpose)
        segment['gaze'] = torch.stack(gaze)
        segment['pose'] = torch.stack(pose)
        segment['bite'] = torch.stack(bite)
        
        return segment

            





    