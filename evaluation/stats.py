from utils.dataset import get_loaders
from collections import defaultdict
import torch
from prettytable import PrettyTable


def bite_preprocess(batch):
    bite = batch['bite']
    bz = bite.size(0)
    bite_reshaped = bite.reshape(bz, 3, 12, -1) # (bz*3, 6, 180)
    bite_sum = bite_reshaped.sum(dim=-1) # (bz, 3, 180)
    bite_tranformed = (bite_sum >= 1).float().unsqueeze(-1).reshape(-1, 1) 

    return bite_tranformed


def speaker_preprocess(batch):
    speaker = batch['speaker']
    bz = speaker.size(0)
    speaker_reshaped = speaker.reshape(bz, 3, 12, -1) # (bz, 3, 6, 180)
    speaker_sum = speaker_reshaped.sum(dim=-1) # (bz, 3, 180)
    speaker_tranformed = (speaker_sum > 0.3 * 90).float().unsqueeze(-1).reshape(-1, 1) 

    return speaker_tranformed

def count(batch, bite_count, speaker_count, name):
    bite = bite_preprocess(batch)
    speaker = speaker_preprocess(batch)
    
    bite_0 = bite[bite == 0].size(0)
    bite_1 = bite[bite == 1].size(0)

    speaker_0 = speaker[speaker == 0].size(0)
    speaker_1 = speaker[speaker == 1].size(0)

    bite_count[name][0] += bite_0
    bite_count[name][1] += bite_1
    speaker_count[name][0] += speaker_0
    speaker_count[name][1] += speaker_1

def print_statistics(bite_count, speaker_count):
    # Create a PrettyTable object
    table = PrettyTable()
    
    # Define the table columns
    table.field_names = ["Task", "Dataset", "0s", "1s", "Total"]

    # Add bite count data to the table
    for dataset in ['train', 'test']:
        zeros = bite_count[dataset][0]
        ones = bite_count[dataset][1]
        total = zeros + ones
        table.add_row(["Bite", dataset.capitalize(), zeros, ones, total])

    # Add speaker count data to the table
    for dataset in ['train', 'test']:
        zeros = speaker_count[dataset][0]
        ones = speaker_count[dataset][1]
        total = zeros + ones
        table.add_row(["Speaker", dataset.capitalize(), zeros, ones, total])

    # Print the table
    print(table)


# stats for bite and speaking classification tasks
def stats():
    train_loader, val_loader, test_loader = get_loaders(batch_path='/home/tangyimi/masked-social-signals/dining_dataset/batch_window36_stride18_v4', 
                                        validation_idx=30, 
                                        batch_size=32, 
                                        num_workers=2)
    
    bite_count = {'train': defaultdict(int), 'test': defaultdict(int)}
    speaker_count = {'train': defaultdict(int), 'test': defaultdict(int)}

    for batch in train_loader:
        count(batch, bite_count, speaker_count, 'train')

    for batch in val_loader:
        count(batch, bite_count, speaker_count, 'train')

    for batch in test_loader:
        count(batch, bite_count, speaker_count, 'test')

    
    print_statistics(bite_count, speaker_count)

if __name__ == '__main__':
    stats()
