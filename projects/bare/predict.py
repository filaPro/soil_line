import os
import gdal
import torch
import numpy as np
import skimage.filters
from argparse import ArgumentParser
from albumentations.pytorch import ToTensorV2

from filter import filter
from train import BaseModel
from dataset import read_tif_files


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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--step', type=int, default=256)
    parser.add_argument('--quantile', type=float, default=.05)
    parser.add_argument('--image-path', default='/data/soil_line/unusable/CH')
    parser.add_argument('--model-path', default='/data/logs/bare/.../checkpoints/....ckpt')
    options = parser.parse_args()
    size = options.size

    out_path = os.path.join(os.path.dirname(options.image_path), 'masks')
    os.makedirs(out_path, exist_ok=True)

    model = BaseModel.load_from_checkpoint(options.model_path)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    weight = get_weight(size)
    base_file_names = sorted(set('_'.join(file_name.split('_')[:4]) for file_name in os.listdir(options.image_path)))
    for base_file_name in base_file_names:
        print(base_file_name)
        images, transform, reference = read_tif_files(options.image_path, base_file_name)
        image = np.stack(tuple(images.values()), axis=-1)
        grid = get_grid(image.shape[1], image.shape[0], size, options.step)
        mask = np.zeros((image.shape[0], image.shape[1]))
        counts = np.zeros((image.shape[0], image.shape[1]))
        for grid_batch in np.split(grid, np.arange(0, len(grid), options.batch_size)[1:]):
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
        mask = filter(mask, images, options.quantile)
        save(
            mask,
            os.path.join(out_path, f'{base_file_name}.tif'),
            reference,
            transform
        )

    # import gdal
    # import skimage.io
    # mask = gdal.Open(f'/data/soil_line/bare/open_soil/{base_file_name}_mask.tif')
    # mask = mask.GetRasterBand(1).ReadAsArray()
    # skimage.io.imsave(
    #     os.path.join(os.path.dirname(os.path.dirname(options.model_path)), 'masks', f'{base_file_name}_.png'),
    #     (mask * 255).astype(np.uint8)
    # )
    # mask = gdal.Open(os.path.join(out_path, f'{base_file_name}.tif'))
    # mask = mask.GetRasterBand(1).ReadAsArray()
    # skimage.io.imsave(
    #     os.path.join(os.path.join(out_path, f'{base_file_name}__.png')),
    #     (mask * 255).astype(np.uint8)
    # )
