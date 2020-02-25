import os
from functools import partial
import tensorflow as tf

from utils import N_PROCESSES


def parse_example(example, n_channels):
    feature = {
        'width': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'x_min': tf.io.FixedLenFeature((), tf.float32),
        'y_max': tf.io.FixedLenFeature((), tf.float32),
        'positive': tf.io.VarLenFeature(tf.int64),
        'negative': tf.io.VarLenFeature(tf.int64),
    }
    for i in range(n_channels):
        feature[f'channel/{i}'] = tf.io.VarLenFeature(tf.float32)
    data = tf.io.parse_single_example(example, feature)
    keys = [f'channel/{i}' for i in range(n_channels)] + ['positive', 'negative']
    for key in keys:
        data[key] = tf.sparse.to_dense(data[key])
    return data


def list_tfrecords(path):
    return list(os.path.join(path, name) for name in os.listdir(path))


def make_class_dataset(paths, transform_lambda, label, n_processes=N_PROCESSES):
    return tf.data.Dataset.from_tensor_slices(
        paths
    ).interleave(
        tf.data.TFRecordDataset,
        cycle_length=n_processes,
        num_parallel_calls=n_processes
    ).map(
        parse_example,
        num_parallel_calls=n_processes
    ).map(
        partial(transform_lambda, label=label),
        num_parallel_calls=n_processes
    ).filter(
        lambda x: x['status']
    ).repeat(
    )


def make_dataset(paths, transform_lambda, batch_size, buffer_size, n_processes=N_PROCESSES):
    positive = make_class_dataset(paths, transform_lambda, 'positive')
    negative = make_class_dataset(paths, transform_lambda, 'negative')
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
    image = tf.concat((image, mask), axis=-1)
    half_size = tf.cast(size / 2, tf.int64)
    left_pad = -tf.minimum(x - half_size, 0)
    top_pad = -tf.minimum(y - half_size, 0)
    right_pad = -tf.minimum(image.shape[1] - (left_pad + size), 0)
    bottom_pad = -tf.minimum(image.shape[0] - (top_pad + size), 0)
    image = tf.pad(image, [[top_pad, bottom_pad], [left_pad, right_pad]])
    return tf.image.crop_to_bounding_box(image, top_pad + y - half_size, left_pad + x - half_size, size, size)



def transform(data, label, masks, xs, ys, n_channels, size, resolution):
    channels = [tf.reshape(data[f'channels/{i}'], (data['height'], data['width'])) for i in range(n_channels)]
    n_fields = len(data[label])
    if not n_fields:
        return {'status': False}
    i = tf.random.uniform((), 0, n_fields, dtype=tf.int64)
    mask = tf.io.decode_png(masks[i], channels=1)
    image = tf.concat(
        [crop_or_pad(
            image=channels[i],
            x=tf.cast((xs[i] - data['x_min']) / resolution, tf.int64), 
            y=tf.cast((data['y_max'] - ys[i]) / resolution, tf.int64),
            size=size
        ) for i in range(n_channels)] +
        [crop_or_pad(
            image=tf.cast(mask, tf.float32),
            x=tf.cast((mask.shape[1] / 2, tf.int64) / resolution, tf.int64),
            y=tf.cast((mask.shape[0] / 2, tf.int64) / resolution, tf.int64),
            size=size
        )] + 
        [crop_or_pad(
            image=tf.ones_like(mask, tf.float32),
            x=tf.cast((mask.shape[1] / 2, tf.int64) / resolution, tf.int64),
            y=tf.cast((mask.shape[0] / 2, tf.int64) / resolution, tf.int64),
            size=size
        )],
        axis=0
    )
    return {
        'image': image,
        'label': label == 'positive'
    }

