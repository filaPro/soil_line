import os
import numpy as np
import tensorflow as tf
from osgeo import gdal
from argparse import ArgumentParser
from classification_models.tfkeras import Classifiers

from utils import RESOLUTION, list_tif_files, list_channels


def make_generator(base_file_names, shape_path, names):
    masks = read_masks(shape_path)
    yield image


def run(base_file_names, shape_path, out_path, image_size, batch_size):
    names = []
    images = make_generator(base_file_names, shape_path, names)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(size, size, 6 + 1)),
        tf.keras.layers.Lambda(lambda x: tf.stack((
            x[..., 1], x[..., 2], x[..., -1]
        ), axis=-1) * 255.),
        Classifiers.get('resnet18')[0](input_shape=(size, size, 3), weights='imagenet', include_top=False),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    probabilities = model.predict(images)

     




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/data')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable/tmp-resnet-18')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=224) 
    options = vars(parser.parse_args())

    tif_path=os.path.join(options['in_path'], 'CH')
    base_file_names = tuple(file_name for file_name in list_tif_files(tif_path) if '173' in file_name)

    run(
        base_file_names=base_file_names,
        shape_path=os.path.join(options['in_path'], 'fields.shp')
        out_path=options['out_path']
        image_size=options['image_size']
        batch_size=options['batch_size']
    )

