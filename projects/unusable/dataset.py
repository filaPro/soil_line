import os
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


def make_class_dataset(paths, transform_lambda, label, batch_size, buffer_size, n_processes=N_PROCESSES):
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
    )repeat(
    )


def make_dataset(paths, transform_lambda, batch_size, buffer_size):
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


def transform(data, label, masks, xs, ys, n_channels):
    channels = [tf.reshape(data[f'channels/{i}'], (data['height'], data['width'])) for i in range(n_channels)]
    image = tf.concat(channels, axis=0)
    n_fields = len(data[label])
    if not n_fields:
        return {'status': False}
    i = tf.random.uniform((), 0, n_fields, dtype=tf.int64)
    mask = tf.io.decode_png(masks[i], channels=1)
    
