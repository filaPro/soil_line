import os
import numpy as np
import pandas as pd
import tensorflow as tf
from osgeo import gdal
from datetime import datetime
from argparse import ArgumentParser
from classification_models.tfkeras import Classifiers

from utils import RESOLUTION, list_tif_files, list_channels, read_masks


def find_latest_weights(path):
    return sorted(filter(lambda s: 'weights' in s, os.listdir(path)))[-1]


def pad(image, size):
    half_size = size // 2
    return np.pad(image, ((half_size, half_size), (half_size, half_size)))


def crop(image, x, y, size):
    half_size = size // 2
    return image[y - half_size: y + half_size, x - half_size: x + half_size]


def make_generator(base_file_names, shape_path, names, size, resolution=RESOLUTION):
    masks = read_masks(shape_path)
    for name in masks.keys():
        masks[name]['mask'] = pad(masks[name]['mask'], size)
    for i, base_file_name in enumerate(base_file_names):
        print(f'{datetime.now()} {i}/{len(base_file_names)} {base_file_name}')
        channels = []
        for tif_file_name in list_channels(base_file_name):
            tif_file = gdal.Open(os.path.join(tif_path, tif_file_name))
            x_min, _, _, y_max, _, _ = tif_file.GetGeoTransform()
            channels.append(tif_file.GetRasterBand(1).ReadAsArray())
            x_max = x_min + resolution * channels[-1].shape[1]
            y_min = y_max - resolution * channels[-1].shape[0]
            channels[-1] = pad(channels[-1], size)
        for name in masks.keys():
            if x_min < masks[name]['x'] < x_max and y_min < masks[name]['y'] < y_max:
                names.append((base_file_name, name))
                yield np.stack(
                    [crop(
                        image=channel,
                        x=int((masks[name]['x'] - x_min) / resolution) + size // 2,
                        y=int((y_max - masks[name]['y']) / resolution) + size // 2,
                        size=size
                    ) for channel in channels] +
                    [crop(
                        image=masks[name]['mask'].astype(np.float32),
                        x=masks[name]['mask'].shape[1] // 2,
                        y=masks[name]['mask'].shape[0] // 2,
                        size=size
                    )],
                    axis=-1
                )


def batch_generator(generator, batch_size):
    batch = []
    for item in generator:
        if len(batch) < batch_size:
            batch.append(item)
        else:
            yield np.array(batch)
            batch = [item]
    if len(batch) > 0:
        yield np.array(batch)


def run(base_file_names, shape_path, out_path, size, batch_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(size, size, 6 + 1)),
        tf.keras.layers.Lambda(lambda x: tf.stack((
            x[..., 1], x[..., 2], x[..., -1]
        ), axis=-1) * 255.),
        Classifiers.get('resnet18')[0](input_shape=(size, size, 3), weights=None, include_top=False),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    weights = find_latest_weights(out_path)
    print(f'loading checkpoint: {weights}')
    model.load_weights(os.path.join(out_path, weights))

    names = []
    generator = batch_generator(make_generator(base_file_names, shape_path, names, size), batch_size)
    probabilities = []
    for batch in generator:
        probabilities += model.predict_on_batch(batch).numpy()[:, 0].tolist()
    assert len(names) == len(probabilities)
    image_names, mask_names = {}, {}
    for image_name, mask_name in names:
        if not image_name in image_names:
            image_names[image_name] = len(image_names)
        if not mask_name in mask_names:
            mask_names[mask_name] = len(mask_names)
    values = np.zeros((len(image_names), len(mask_names)))
    for probability, (image_name, mask_name) in zip(probabilities, names):
        values[image_names[image_name], mask_names[mask_name]] = probability
    pd.DataFrame(data=values, index=image_names, columns=mask_names).to_csv(
        os.path.join(out_path, 'out.csv'), float_format='%.3f'
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/data')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable/tmp-resnet-18')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--image-size', type=int, default=224) 
    options = vars(parser.parse_args())

    tif_path=os.path.join(options['in_path'], 'CH')
    run(
        base_file_names=tuple(file_name for file_name in list_tif_files(tif_path) if '173' in file_name),
        shape_path=os.path.join(options['in_path'], 'fields.shp'),
        out_path=options['out_path'],
        size=options['image_size'],
        batch_size=options['batch_size'],
    )
