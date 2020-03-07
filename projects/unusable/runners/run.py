import os
import numpy as np
import tensorflow as tf
from functools import partial
from datetime import datetime
from argparse import ArgumentParser

from utils import read_masks
from dataset import list_tfrecords, make_dataset, transform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/volume/unusable_tfrecords_v2')
    parser.add_argument('--shape-path', type=str, default='/data/fields.shp')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--resolution', type=float, default=30.)
    parser.add_argument('--image-size', type=int, default=512)
    options = vars(parser.parse_args())

    size = options['image_size']
    out_path = os.path.join(options['out_path'], datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-174-to-173')
    os.mkdir(out_path)
    training_paths = list_tfrecords(options['in_path'], '174')
    validation_paths = list_tfrecords(options['in_path'], '173')
    assert len(training_paths) + len(validation_paths) == len(list_tfrecords(options['in_path'], ''))

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
        tf.keras.layers.Input(shape=(size, size, 6 + 1)),
        tf.keras.layers.Conv2D(32, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(128, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(256, 3, 2, activation='relu'),
        tf.keras.layers.Conv2D(512, 3, 2, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.PrecisionAtRecall(.5, name='pre/.5rec'),
            tf.keras.metrics.PrecisionAtRecall(.9, name='pre/.9rec'),
            tf.keras.metrics.PrecisionAtRecall(.99, name='pre/.99rec')
        ]
    )
    history = model.fit_generator(
        training_dataset,
        steps_per_epoch=25,
        epochs=100,
        validation_data=validation_dataset,
        validation_steps=5,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(out_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                verbose=1,
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=.2,
                patience=5
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=out_path,
                write_graph=False,
                profile_batch=0
            ),
            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(out_path, 'log.csv'),
                append=True
            )
        ]
    )
