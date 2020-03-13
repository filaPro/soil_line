import os
from functools import partial
import tensorflow as tf

from utils import N_PROCESSES, RESOLUTION


def parse_example(example, n_channels):
    feature = {
        'width': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'x_min': tf.io.FixedLenFeature((), tf.float32),
        'y_max': tf.io.FixedLenFeature((), tf.float32),
        'positive': tf.io.VarLenFeature(tf.int64),
        'negative': tf.io.VarLenFeature(tf.int64),
        'day': tf.io.FixedLenFeature((), tf.int64),
        'satellite': tf.io.FixedLenFeature((), tf.int64)
    }
    for i in range(n_channels):
        feature[f'channels/{i}'] = tf.io.VarLenFeature(tf.float32)
    data = tf.io.parse_single_example(example, feature)
    keys = [f'channels/{i}' for i in range(n_channels)] + ['positive', 'negative']
    for key in keys:
        data[key] = tf.sparse.to_dense(data[key])
    return data


def list_tfrecords(path, name):
    return list(os.path.join(path, file_name) for file_name in os.listdir(path) if name in file_name)


def make_class_dataset(paths, transform_lambda, n_channels, label, n_processes=N_PROCESSES):
    return tf.data.Dataset.from_tensor_slices(
        paths
    ).interleave(
        tf.data.TFRecordDataset,
        cycle_length=n_processes,
        num_parallel_calls=n_processes
    ).map(
        partial(parse_example, n_channels=n_channels),
        num_parallel_calls=n_processes
    ).filter(
        lambda x: len(x[label]) > 0
    ).map(
        partial(transform_lambda, label=label),
        num_parallel_calls=n_processes
    ).repeat(
    )


def make_dataset(paths, transform_lambda, n_channels, batch_size, buffer_size, n_processes=N_PROCESSES):
    positive = make_class_dataset(paths, transform_lambda, n_channels, 'positive')
    negative = make_class_dataset(paths, transform_lambda, n_channels, 'negative')
    return tf.data.experimental.sample_from_datasets(
        (positive, negative)
    ).shuffle(
        buffer_size
    ).batch(
        batch_size
    ).prefetch(
        n_processes
    )


def crop_or_pad(image, x, y, size):
    half_size = tf.cast(size / 2, tf.int32)
    left_pad = -tf.minimum(x - half_size, 0)
    top_pad = -tf.minimum(y - half_size, 0)
    right_pad = -tf.minimum(tf.shape(image)[1] - x - half_size, 0)
    bottom_pad = -tf.minimum(tf.shape(image)[0] - y - half_size, 0)
    image = tf.pad(image, [[top_pad, bottom_pad], [left_pad, right_pad]])
    return tf.image.crop_to_bounding_box(
        image[..., tf.newaxis], top_pad + y - half_size, left_pad + x - half_size, size, size
    )[..., 0]


def transform(data, label, masks, xs, ys, n_channels, size, resolution=RESOLUTION):
    channels = [tf.reshape(data[f'channels/{i}'], (data['height'], data['width'])) for i in range(n_channels)]
    index = tf.random.uniform((), 0, len(data[label]), dtype=tf.int32)
    mask = tf.io.decode_png(masks[index], channels=1)[..., 0]
    image = tf.stack(
        [crop_or_pad(
            image=channel,
            x=tf.cast((xs[index] - data['x_min']) / resolution, tf.int32),
            y=tf.cast((data['y_max'] - ys[index]) / resolution, tf.int32),
            size=size
        ) for channel in channels] +
        [crop_or_pad(
            image=tf.cast(mask, tf.float32),
            x=tf.cast(tf.shape(mask)[1] / 2, tf.int32),
            y=tf.cast(tf.shape(mask)[0] / 2, tf.int32),
            size=size
        )],
        axis=-1
    )
    return image, label == 'positive'
