import random, os
import numpy as np
import torch
import wandb
import pandas as pd
from scipy.interpolate import interpn

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


class MetricTracker:
    def __init__(self, keys, log_data=False, mode='train'):
        self.mode = mode
        self.log_data = log_data
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, step, n=1):
        if self.log_data:
            wandb.log({key:value, self.mode+'_step':step})
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self, epoch):
        temp_log = dict(self._data.average)
        log = {}
        for key in temp_log.keys():
            log[key.replace('batch', 'epoch')] = temp_log[key]

        return_dict = log.copy()
        log['epoch'] = epoch
        if self.log_data:
            wandb.log(log)
        return return_dict

def smoothing(batch, batch_length):
    bz = batch.size(0)
    batch = batch.cpu().numpy()
    points = np.arange(batch_length)
    xi = points[::2]

    result = []

    for batch_idx in range(bz):
        for person_idx in range(3):
            xy_values = []
            for feature_idx in range(0, 26, 2):
                current = batch[batch_idx, person_idx, :, feature_idx:feature_idx+2]
                smoothed = interpn(points=(points,), values=current, xi=xi, method='cubic')
                xy_values.append(smoothed)

            person_result = np.concatenate(xy_values, axis=-1)
            result.append(person_result)
    
    return torch.tensor(result).view(bz, 3, batch_length // 2, -1)

