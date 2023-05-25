import random, os
import numpy as np
import torch
import wandb
import pandas as pd

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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