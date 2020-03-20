import os
import numpy as np
import pandas as pd
from osgeo import gdal
import tensorflow as tf
from collections import defaultdict

from utils import RESOLUTION, read_fields, list_tif_files


def list_channels(base_file_name):
    channels = {
        'blue': ['01', '02'],
        'green': ['02', '03'],
        'red': ['03', '04'],
        'nir': ['04', '05'],
        'swir1': ['05', '06'],
        'swir2': ['07', '07']
    }
    channel_shift = base_file_name.split('_')[2][-1] == '8'
    return {channel: f'{base_file_name}_{channel}_{channels[channel][channel_shift]}.tif' for channel in channels}


def concatenate(items, aggregation):
    result = defaultdict(list)
    for item in items:
        for key, value in item.items():
            result[key].append(value)
    for key in result:
        result[key] = aggregation(result[key])
    return result


def get_intersecting_field_names(fields, x_min, y_min, x_max, y_max):
    names = []
    for name in fields:
        if x_min < fields[name]['x'] < x_max and y_min < fields[name]['y'] < y_max:
            names.append(name)
    return names


def read_tif_file(path):
    tif_file = gdal.Open(path)
    x_min, _, _, y_max, _, _ = tif_file.GetGeoTransform()
    image = tif_file.GetRasterBand(1).ReadAsArray()
    return image, x_min, y_max


def read_tif_files(path, base_file_name, resolution=RESOLUTION):
    images = {}
    for channel, file_name in list_channels(base_file_name).items():
        image, x_min, y_max = read_tif_file(os.path.join(path, file_name))
        images[channel] = image
    x_max = x_min + image.shape[1] * resolution
    y_min = y_max - image.shape[0] * resolution
    return images, x_min, y_min, x_max, y_max


class TrainingSequence(tf.keras.utils.Sequence):
    def __init__(
        self, tif_path, base_file_names, fields, data_frame, n_batch_images, n_batch_fields, transform_lambda
    ):
        super().__init__()
        self.tif_path = tif_path
        self.base_file_names = base_file_names
        self.fields = fields
        self.data_frame = data_frame
        self.n_batch_images = n_batch_images
        self.n_batch_fields = n_batch_fields
        self.transform_lambda = transform_lambda

    def __len__(self):
        return 1

    def __getitem__(self, _):
        results = []
        for _ in range(self.n_batch_images):
            base_file_name = np.random.choice(self.base_file_names)
            images, x_min, y_min, x_max, y_max = read_tif_files(self.tif_path, base_file_name)
            positive_names = self.data_frame[self.data_frame['NDVI_map'] == f'{base_file_name}_n.tif']['name'].values
            positive_names = list(set(positive_names) & set(self.fields.keys()))  # TODO: temporary bug in dataset
            intersecting_names = get_intersecting_field_names(self.fields, x_min, y_min, x_max, y_max)
            negative_names = list(set(intersecting_names) - set(positive_names))
            for _ in range(self.n_batch_fields if len(positive_names) > 0 else 1):  # TODO: strange balancing
                names, label = (positive_names, True) if np.random.uniform() < .75 and len(positive_names) > 0\
                    else (negative_names, False)
                field_name = np.random.choice(names)
                results.append(self.transform_lambda(
                    images=images,
                    base_file_name=base_file_name,
                    x_min=x_min,
                    y_max=y_max,
                    mask=self.fields[field_name]['mask'],
                    field_name=field_name,
                    x=self.fields[field_name]['x'],
                    y=self.fields[field_name]['y'],
                    label=label
                ))
        return concatenate(results, np.stack)
        

class TestSequence(tf.keras.utils.Sequence):
    def __init__(self, tif_path, base_file_names, fields, n_batch_fields, transform_lambda):
        super().__init__()
        self.tif_path = tif_path
        self.base_file_names = base_file_names
        self.fields = fields
        self.n_batch_fields = n_batch_fields
        self.transform_lambda = transform_lambda

    def __len__(self):
        return len(self.base_file_names) * ceil(len(self.fields) / self.n_batch_fields)

    def __getitem__(self, index):
        results = []
        base_file_name = base_file_names(index / ceil(len(self.fields) / self.n_batch_fields))
        images, x_min, y_min, x_max, y_max = read_tif_files(self.tif_path, base_file_name)
        min_field_index = base_file_names(index % ceil(len(self.fields) / self.n_batch_fields))
        max_field_index = min(min_field_index + self.n_batch_fields, len(self.fields))
        intersecting_names = get_intersecting_field_names(self.fields, x_min, y_min, x_max, y_max)
        names = list(set(intersecting_names) & set(list(fields.keys())[min_field_index: max_field_index]))
        for field_name in names:
            results.append(self.transform_lambda(
                images=images,
                base_file_name=base_file_name,
                x_min=x_min,
                y_max=y_max,
                mask=self.fields[field_name]['mask'],
                field_name=field_name,
                x=self.fields[field_name]['x'],
                y=self.fields[field_name]['y'],
                label=label
            ))
        return results


        

