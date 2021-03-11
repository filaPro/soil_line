import os

import gdal
import numpy as np
import albumentations
import pytorch_lightning
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, ConcatDataset

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

    # Here we warp Sentinel images to have the same resolution as Landsat ones.
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

    # If the shapes of images are not the same we crop them my minimal.
    shapes = tuple(image.shape for image in images.values())
    if len(shapes) != 1:
        height, width = np.min(shapes, axis=0)
        for channel in images.keys():
            if images[channel].shape != (height, width):
                images[channel] = images[channel][:height, :width]

    return images, transform, reference


class BaseDataset(Dataset):
    def __init__(self, image_path, mask_path, augmentation):
        self.image_path = image_path
        self.mask_path = mask_path
        self.augmentation = augmentation
        self.base_file_names = tuple(map(lambda x: os.path.basename(x)[:-9], os.listdir(self.mask_path)))

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


class BaseDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, n_repeats, batch_size, n_workers, size,
                 training_image_paths, training_mask_paths, validation_image_paths, validation_mask_paths):
        super().__init__()
        self.n_repeats = n_repeats
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.size = size
        self.training_image_paths = training_image_paths
        self.training_mask_paths = training_mask_paths
        self.validation_image_paths = validation_image_paths
        self.validation_mask_paths = validation_mask_paths

    def _make_dataloader(self, image_paths, mask_paths, augmentation):
        assert len(image_paths) == len(mask_paths)
        return DataLoader(
            RepeatedDataset(
                ConcatDataset([
                    BaseDataset(
                        image_path=image_path,
                        mask_path=mask_path,
                        augmentation=augmentation
                    ) for image_path, mask_path in zip(image_paths, mask_paths)
                ]),
                n_repeats=self.n_repeats
            ),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            worker_init_fn=lambda x: np.random.seed(x)
        )

    def train_dataloader(self):
        return self._make_dataloader(
            image_paths=self.training_image_paths,
            mask_paths=self.training_mask_paths,
            augmentation=albumentations.Compose([
                albumentations.RandomRotate90(),
                albumentations.RandomCrop(self.size, self.size),
                ToTensorV2(),
            ])
        )

    def val_dataloader(self):
        return self._make_dataloader(
            image_paths=self.validation_image_paths,
            mask_paths=self.validation_mask_paths,
            augmentation=albumentations.Compose([
                albumentations.RandomCrop(self.size, self.size),
                ToTensorV2(),
            ])
        )

    def test_dataloader(self):
        return self.val_dataloader()
