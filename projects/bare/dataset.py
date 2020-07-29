import os
import gdal
import numpy as np
from torch.utils.data import Dataset

RESOLUTION = 30.


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


def read_tif_file(path, resolution=RESOLUTION):
    tif_file = gdal.Open(path)
    _, x_resolution, _, _, _, y_resolution = tif_file.GetGeoTransform()
    if x_resolution != resolution or y_resolution != -resolution:
        tif_file = gdal.Warp(
            destNameOrDestDS='',
            srcDSOrSrcDSTab=tif_file,
            format='VRT',
            xRes=resolution,
            yRes=resolution,
            resampleAlg='cubic'
        )
    return tif_file.GetRasterBand(1).ReadAsArray(), tif_file.GetGeoTransform(), tif_file.GetSpatialRef()


def read_tif_files(path, base_file_name):
    images = {}
    transform, reference = None, None
    for channel, file_name in list_channels(base_file_name).items():
        image, transform, reference = read_tif_file(os.path.join(path, file_name))
        images[channel] = image
    return images, transform, reference


class BaseDataset(Dataset):
    def __init__(self, image_path, mask_paths, augmentation):
        self.image_path = image_path
        self.mask_path = os.path.dirname(mask_paths[0])
        self.augmentation = augmentation
        self.base_file_names = tuple(map(lambda x: os.path.basename(x)[:-9], mask_paths))

    def __len__(self):
        return len(self.base_file_names)

    def __getitem__(self, item):
        base_file_name = self.base_file_names[item]
        images = read_tif_files(self.image_path, base_file_name)[0]
        image = np.stack(tuple(images.values()), axis=-1)
        mask = read_tif_file(os.path.join(self.mask_path, f'{base_file_name}_mask.tif'))[0].astype(np.uint8)
        augmented = self.augmentation(image=image, mask=mask)
        return augmented['image'], augmented['mask']


class RepeatedDataset(Dataset):
    def __init__(self, dataset, n_repeats):
        self.dataset = dataset
        self.n_repeats = n_repeats

    def __len__(self):
        return len(self.dataset) * self.n_repeats

    def __getitem__(self, item):
        return self.dataset[item % len(self.dataset)]
