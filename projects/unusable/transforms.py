import scipy.stats
import numpy as np

from utils import RESOLUTION


def crop_or_pad(image, x, y, size):
    assert size % 2 == 0
    half_size = size // 2
    crop = image[
        max(y - half_size, 0): min(y + half_size, image.shape[0]),
        max(x - half_size, 0): min(x + half_size, image.shape[1]),
    ]
    return np.pad(crop, (
        (-min(y - half_size, 0), -min(image.shape[0] - y - half_size, 0)),
        (-min(x - half_size, 0), -min(image.shape[1] - x - half_size, 0))
    ))


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
    raise ValueError(f'Unknown satellite for file: {base_file_name}')


def catboost_transform(images, base_file_name, x_min, y_max, mask, field_name, x, y, label, size, resolution=RESOLUTION):
    cropped_images = {
        name: crop_or_pad(image, int((x - x_min) / resolution), int((y_max - y) / resolution), size)
        for name, image in images.items()
    }
    cropped_images['ndvi'] = (cropped_images['nir'] - cropped_images['red']) / \
        (cropped_images['nir'] + cropped_images['red'] + .0001)
    cropped_mask = crop_or_pad(mask, mask.shape[1] // 2, mask.shape[0] // 2, size)
    values = {name: image[cropped_mask > 0] for name, image in cropped_images.items()}
    results = {
        'day': get_day(base_file_name),
        'satellite': get_satellite(base_file_name),
        'label': label,
        'file_name': base_file_name,
        'field_name': field_name
    }
    for name, value in values.items():
        for quantile in (.01, .05, .10, .5, .9, .95, .99):
            results[f'{name}_p_{quantile}'] = np.quantile(value, quantile)
        results[f'{name}_std'] = np.std(value)
        results[f'{name}_skew'] = scipy.stats.skew(value)
        results[f'{name}_kurtosis'] = scipy.stats.kurtosis(value)
    return results

