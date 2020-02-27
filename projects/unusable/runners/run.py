import numpy as np
import tensorflow as tf
from functools import partial
from argparse import ArgumentParser

from utils import read_masks
from dataset import list_tfrecords, make_dataset, transform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/volume/unusable_tfrecords')
    parser.add_argument('--shape-path', type=str, default='/data/fields.shp')
    parser.add_argument('--out-path', type=str, default='/volume/logs/tmp')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--resolution', type=float, default=30.)
    options = vars(parser.parse_args())

    size = 512
    paths = list_tfrecords(options['in_path'])
    n_training_paths = int(len(paths) * .8)
    training_paths = paths[:n_training_paths]
    validation_paths = paths[n_training_paths:]

    masks_data = read_masks(
        shape_path=options['shape_path'],
        resolution=options['resolution']
    )
    masks, xs, ys = [], [], []
    for _, item in masks_data.items():
        masks.append(tf.image.encode_png(item['mask'][..., np.newaxis]))
        xs.append(item['x'])
        ys.append(item['y'])
    transform_lambda = partial(
        transform,
        masks=tf.stack(masks),
        xs=tf.cast(tf.stack(xs), tf.float32),
        ys=tf.cast(tf.stack(ys), tf.float32),
        n_channels=6,
        size=size,
        resolution=options['resolution']
    )
    dataset_lambda = partial(
        make_dataset,
        transform_lambda=transform_lambda,
        n_channels=6,
        batch_size=options['batch_size'],
        buffer_size=8,
    )
    training_dataset = dataset_lambda(paths=training_paths)
    validation_dataset = dataset_lambda(paths=validation_paths)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(size, size, 6 + 2)),
        tf.keras.layers.Conv2D(32, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(128, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(256, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(512, 3, 2, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    history = model.fit_generator(
        training_dataset,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_dataset,
        validation_steps=100
    )
