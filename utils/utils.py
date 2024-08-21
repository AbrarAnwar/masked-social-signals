import numpy as np
import torch
import pandas as pd
from scipy.interpolate import interpn


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def smoothing(pose_segment):
    pose_segment = np.array(pose_segment)
    batch_length = pose_segment.shape[1]
    points = np.arange(batch_length)
    xi = points[::2]

    result = []

    for person_idx in range(3):
        xy_values = []
        for feature_idx in range(0, 26, 2):
            current = pose_segment[person_idx, :, feature_idx:feature_idx+2]
            smoothed = interpn(points=(points,), values=current, xi=xi, method='cubic')
            xy_values.append(smoothed)

        person_result = np.concatenate(xy_values, axis=-1)
        result.append(person_result)
    
    return np.array(result)


def get_search_hparams(config):
    search_hparams = []
    for k,v in config['parameters'].items():
        if 'values' in v:
            search_hparams.append(k)
    return search_hparams


def get_experiment_name(search_hparams, hparams):
    if search_hparams:
        return hparams['model'] + '_'.join([f'{k}={v}' for k,v in hparams.items() if k in search_hparams])
    return f"default_{hparams['model']}"

