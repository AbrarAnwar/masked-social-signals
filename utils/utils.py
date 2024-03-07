import random, os
import numpy as np
import torch
import wandb
import pandas as pd
from scipy.interpolate import interpn


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


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

