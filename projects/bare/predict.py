import json
import os
import sys
import time
from argparse import ArgumentParser
from types import SimpleNamespace

import gdal
import numpy as np
import skimage.filters
import torch
from albumentations.pytorch import ToTensorV2

from dataset import read_tif_files
from filter import filter
from train import BaseModel


def get_weight(size):
    weight = np.zeros((size, size))
    weight[size // 2, size // 2] = 1
    return skimage.filters.gaussian(weight * size * size, sigma=size / 7)


def get_grid(width, height, size, step):
    assert width >= size and height >= size
    grid = []
    for y in range(height // step):
        for x in range(width // step):
            grid.append([
                y * step - max(y * step + size - height, 0),
                x * step - max(x * step + size - width, 0)
            ])
    return np.array(grid)


def save(image, path, reference, transform):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        utf8_path=path,
        xsize=image.shape[1],
        ysize=image.shape[0],
        bands=1,
        eType=gdal.GDT_Float32
    )
    dataset.SetGeoTransform(transform)
    dataset.SetProjection(reference.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(image)
    dataset.FlushCache()


def load_proj():
    if getattr(sys, 'frozen', False):  # if we are inside .exe
        os.environ['PROJ_LIB'] = os.path.join(os.path.split(sys.executable)[0], 'osgeo', 'data', 'proj')
    elif sys.platform == 'win32':
        os.environ['PROJ_LIB'] = os.path.join(os.path.split(sys.executable)[0], '..',
                                              'Lib', 'site-packages', 'osgeo', 'data', 'proj')


class Namespace(SimpleNamespace):
    def update(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == '__main__':
    load_proj()

    params = Namespace()
    params.batch_size = 32
    params.size = 512
    params.step = 256
    params.quantile = .05

    parser = ArgumentParser()
    parser.add_argument('--json', type=str, default='soilline_bare_predict_config.json')
    options = vars(parser.parse_args())
    json_path = options['json']

    override = json.load(open(json_path))
    params.update(**override)
    size = params.size

    out_path = os.path.join(os.path.dirname(params.image_path), 'masks')
    os.makedirs(out_path, exist_ok=True)

    model = BaseModel.load_from_checkpoint(params.model_path)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    weight = get_weight(size)
    base_file_names = sorted(set('_'.join(file_name.split('_')[:4]) for file_name in os.listdir(params.image_path)))

    t = time.time()
    print('start')

    for base_file_name in base_file_names:
        print(base_file_name, time.time() - t)
        images, transform, reference = read_tif_files(params.image_path, base_file_name)
        image = np.stack(tuple(images.values()), axis=-1)
        grid = get_grid(image.shape[1], image.shape[0], size, params.step)
        mask = np.zeros((image.shape[0], image.shape[1]))
        counts = np.zeros((image.shape[0], image.shape[1]))
        for grid_batch in np.split(grid, np.arange(0, len(grid), params.batch_size)[1:]):
            batch = []
            for y, x in grid_batch:
                batch.append(ToTensorV2().apply(image[y: y + size, x: x + size]))
            with torch.no_grad():
                masks = torch.sigmoid(model(torch.stack(batch).to(device))).detach().cpu().numpy()
            for i in range(len(masks)):
                y, x = grid_batch[i]
                mask[y: y + size, x: x + size] += masks[i] * weight
                counts[y: y + size, x: x + size] += weight
        assert np.all(counts > 0)
        mask /= counts
        mask = filter(mask, images, params.quantile)
        save(
            mask,
            os.path.join(out_path, f'{base_file_name}.tif'),
            reference,
            transform
        )
