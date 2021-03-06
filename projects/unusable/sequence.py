import os
import time
import numpy as np
from math import ceil
from osgeo import gdal, osr
import tensorflow as tf
from collections import defaultdict

from utils import RESOLUTION


def list_channels(base_file_name):
    channels = {
        'blue': ['01', '02', '02'],
        'green': ['02', '03', '03'],
        'red': ['03', '04', '04'],
        'nir': ['04', '05', '08'],
        'swir1': ['05', '06', '11'],
        'swir2': ['07', '07', '12']
    }
    channel_shift = {
        'LT04': 0,
        'LT05': 0,
        'LE07': 0,
        'LC08': 1,
        'S2AB': 2
    }[base_file_name.split('_')[2]]
    return {channel: f'{base_file_name}_{channel}_{channels[channel][channel_shift]}.tif' for channel in channels}


def aggregate(items, aggregation):
    result = defaultdict(list)
    for item in items:
        for key, value in item.items():
            result[key].append(value)
    for key in result:
        result[key] = aggregation(result[key])
    return result


def keras_aggregate(items):
    result = aggregate(items, np.stack)
    return result['image'], result['label']


def get_intersecting_field_names(fields, x_min, y_min, x_max, y_max):
    names = []
    for name in fields:
        if x_min < fields[name]['x'] < x_max and y_min < fields[name]['y'] < y_max:
            names.append(name)
    return names


def read_tif_file(path, spatial_reference, resolution=RESOLUTION):
    tif_file = gdal.Open(path)
    _, x_resolution, _, _, _, y_resolution = tif_file.GetGeoTransform()
    if (
        str(osr.SpatialReference(wkt=tif_file.GetProjection())) != str(spatial_reference) or
        x_resolution != resolution or
        y_resolution != -resolution
    ):
        tif_file = gdal.Warp(
            destNameOrDestDS='',
            srcDSOrSrcDSTab=tif_file,
            format='VRT',
            dstSRS=spatial_reference,
            dstNodata=0,
            xRes=resolution,
            yRes=resolution,
            resampleAlg='cubic'
        )
    x_min, _, _, y_max, _, _ = tif_file.GetGeoTransform()
    image = tif_file.GetRasterBand(1).ReadAsArray()
    return image, x_min, y_max


def read_tif_files(path, base_file_name, spatial_reference, resolution=RESOLUTION):
    images = {}
    for channel, file_name in list_channels(base_file_name).items():
        image, x_min, y_max = read_tif_file(os.path.join(path, file_name), spatial_reference)
        images[channel] = image
    x_max = x_min + image.shape[1] * resolution
    y_min = y_max - image.shape[0] * resolution
    return images, x_min, y_min, x_max, y_max


class TrainingSequence(tf.keras.utils.Sequence):
    def __init__(
        self, tif_path, base_file_names, fields, spatial_reference, data_frame, n_batch_images, n_batch_fields,
        transformation, aggregation
    ):
        super().__init__()
        self.tif_path = tif_path
        self.base_file_names = base_file_names
        self.fields = fields
        self.spatial_reference = spatial_reference
        self.data_frame = data_frame
        self.n_batch_images = n_batch_images
        self.n_batch_fields = n_batch_fields
        self.transformation = transformation
        self.aggregation = aggregation
        assert not self.n_batch_images % 2

    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        np.random.seed(index + int(time.time()))
        results = []
        n_remaining_positives, n_remaining_negatives = self.n_batch_images // 2, self.n_batch_images // 2
        while n_remaining_positives + n_remaining_negatives > 0:
            base_file_name = np.random.choice(self.base_file_names)
            images, x_min, y_min, x_max, y_max = read_tif_files(self.tif_path, base_file_name, self.spatial_reference)
            positive_names = self.data_frame[self.data_frame['NDVI_map'] == f'{base_file_name}_n.tif']['name'].values
            positive_names = list(set(positive_names) & set(self.fields.keys()))  # TODO: temporary bug in dataset
            intersecting_names = get_intersecting_field_names(self.fields, x_min, y_min, x_max, y_max)
            negative_names = list(set(intersecting_names) - set(positive_names))

            positive_names = np.random.permutation(positive_names)[:min(self.n_batch_fields, n_remaining_positives)].tolist()
            negative_names = np.random.permutation(negative_names)[:min(self.n_batch_fields, n_remaining_negatives)].tolist()
            n_remaining_positives -= len(positive_names)
            n_remaining_negatives -= len(negative_names)
            for name, label in zip(positive_names + negative_names, [1] * len(positive_names) + [0] * len(negative_names)):
                results.append(self.transformation(
                    images=images,
                    base_file_name=base_file_name,
                    x_min=x_min,
                    y_max=y_max,
                    mask=self.fields[name]['mask'],
                    field_name=name,
                    x=self.fields[name]['x'],
                    y=self.fields[name]['y'],
                    label=label
                ))
        return self.aggregation(results) if len(results) else None
        

class TestSequence(tf.keras.utils.Sequence):
    def __init__(
        self, tif_path, base_file_names, fields, spatial_reference, n_batch_fields, transformation, aggregation
    ):
        super().__init__()
        self.tif_path = tif_path
        self.base_file_names = base_file_names
        self.fields = fields
        self.spatial_reference = spatial_reference
        self.n_batch_fields = n_batch_fields
        self.transformation = transformation
        self.aggregation = aggregation

    def __len__(self):
        return len(self.base_file_names) * ceil(len(self.fields) / self.n_batch_fields)

    def __getitem__(self, index):
        results = []
        base_file_name = self.base_file_names[index // ceil(len(self.fields) / self.n_batch_fields)]
        images, x_min, y_min, x_max, y_max = read_tif_files(self.tif_path, base_file_name, self.spatial_reference)
        min_field_index = index % ceil(len(self.fields) / self.n_batch_fields) * self.n_batch_fields
        max_field_index = min(min_field_index + self.n_batch_fields, len(self.fields))
        intersecting_names = get_intersecting_field_names(self.fields, x_min, y_min, x_max, y_max)
        names = list(set(intersecting_names) & set(list(self.fields.keys())[min_field_index: max_field_index]))
        for field_name in names:
            results.append(self.transformation(
                images=images,
                base_file_name=base_file_name,
                x_min=x_min,
                y_max=y_max,
                mask=self.fields[field_name]['mask'],
                field_name=field_name,
                x=self.fields[field_name]['x'],
                y=self.fields[field_name]['y'],
                label=False
            ))
        return self.aggregation(results) if len(results) else None
