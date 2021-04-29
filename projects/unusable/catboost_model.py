import torch
import numpy as np
import scipy.stats


def get_day(base_file_name):
    string = base_file_name.split('_')[1]
    day = int(string[:2]) * 30 + int(string[2:])
    return day


def get_satellite(base_file_name):
    if 'LT04' in base_file_name:
        return 4
    elif 'LT05' in base_file_name:
        return 5
    elif 'LE07' in base_file_name:
        return 7
    elif 'LC08' in base_file_name:
        return 8
    return 0


def catboost_transform(images, label, field_name, base_file_name):
    for key, value in images.items():
        images[key] = value.data[~value.mask]
    nir = images['nir']
    red = images['red']
    images['ndvi'] = (nir - red) / (nir + red + .0001)
    results = {
        'day': get_day(base_file_name),
        'satellite': get_satellite(base_file_name),
        'label': label,
        'field_name': field_name,
        'base_file_name': base_file_name
    }
    for name, value in images.items():
        for quantile in (.01, .05, .10, .5, .9, .95, .99):
            results[f'{name}_p_{quantile}'] = np.quantile(value, quantile)
        results[f'{name}_std'] = np.std(value)
        results[f'{name}_skew'] = scipy.stats.skew(value)
        results[f'{name}_kurtosis'] = scipy.stats.kurtosis(value)
    return results


def batch_to_numpy(batch):
    for key, value in batch.items():
        if type(value) == torch.Tensor:
            batch[key] = value.numpy()
        batch[key] = batch[key][0]
    return batch
