import torch
from scipy.interpolate import interpn
import numpy as np

class Normalizer():
    def __init__(self, loader) -> None:
        self.loader = loader
        self.normalizer = {'headpose': dict(), 'gaze': dict(), 'pose': dict()}
        self._normalize()

    def _normalize(self):
        total_task = len(self.normalizer)

        min_x = torch.full((total_task,), float('inf'))
        max_x = torch.full((total_task,), float('-inf'))

        min_y = torch.full((total_task,), float('inf'))
        max_y = torch.full((total_task,), float('-inf'))

        for batch in self.loader:
            for task_idx, task in enumerate(self.normalizer.keys()):
                current = batch[task]

                x_coord = current[..., ::2]
                y_coord = current[..., 1::2]

                current_min_x = torch.min(x_coord)
                current_max_x = torch.max(x_coord)
                current_min_y = torch.min(y_coord)
                current_max_y = torch.max(y_coord)

                min_x[task_idx] = torch.min(min_x[task_idx], current_min_x)
                max_x[task_idx] = torch.max(max_x[task_idx], current_max_x)
                min_y[task_idx] = torch.min(min_y[task_idx], current_min_y)
                max_y[task_idx] = torch.max(max_y[task_idx], current_max_y)

        for task_idx, task in enumerate(self.normalizer.keys()):
            current_normalizer = {'x': {'min': min_x[task_idx], 'max': max_x[task_idx]},
                                'y': {'min': min_y[task_idx], 'max': max_y[task_idx]}}
            self.normalizer[task] = current_normalizer


    def minmax_normalize(self, tensor, task):
        normalized = tensor.clone()
        x_coords = normalized[..., ::2]  
        y_coords = normalized[..., 1::2]  

        normalized_x_coords = (x_coords - self.normalizer[task]['x']['min']) / (self.normalizer[task]['x']['max'] - self.normalizer[task]['x']['min'])
        normalized_y_coords = (y_coords - self.normalizer[task]['y']['min']) / (self.normalizer[task]['y']['max'] - self.normalizer[task]['y']['min'])

        normalized[..., ::2] = normalized_x_coords
        normalized[..., 1::2] = normalized_y_coords

        return normalized


    def minmax_denormalize(self, tensor, task):
        demoralized = tensor.clone()
        x_coords = demoralized[..., ::2]  
        y_coords = demoralized[..., 1::2]  

        denormalized_x_coords = x_coords * (self.normalizer[task]['x']['max'] - self.normalizer[task]['x']['min']) + self.normalizer[task]['x']['min']
        denormalized_y_coords = y_coords * (self.normalizer[task]['y']['max'] - self.normalizer[task]['y']['min']) + self.normalizer[task]['y']['min']

        demoralized[..., ::2] = denormalized_x_coords
        demoralized[..., 1::2] = denormalized_y_coords

        return demoralized


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

