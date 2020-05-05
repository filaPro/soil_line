import os
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import albumentations
from functools import partial
from datetime import datetime
from argparse import ArgumentParser

from utils import N_PROCESSES, read_fields, list_tif_files
from sequence import TrainingSequence, keras_aggregate
from transforms import keras_transform

from resnet import ResNet18


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/volume/unusable')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable')
    parser.add_argument('--n-training-batches', type=int, default=500)
    parser.add_argument('--n-validation-batches', type=int, default=100)
    parser.add_argument('--n-batch-images', type=int, default=64)
    parser.add_argument('--n-batch-fields', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=128)
    options = vars(parser.parse_args())

    size = options['image_size']
    in_path = options['in_path']
    out_path = os.path.join(options['out_path'], 'tmp')  # datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(out_path, exist_ok=True)
    tif_path = os.path.join(in_path, 'CH')
    training_file_names = list_tif_files(tif_path, '_174')
    validation_file_names = list_tif_files(tif_path, '_173')
    assert len(training_file_names) + len(validation_file_names) == len(list_tif_files(tif_path, ''))

    fields = read_fields(os.path.join(in_path, 'fields.shp'))
    data_frame = pd.read_excel(os.path.join(in_path, 'NDVI_list.xls'))
    build_sequence = partial(
        TrainingSequence,
        tif_path=tif_path,
        fields=fields,
        data_frame=data_frame,
        n_batch_images=options['n_batch_images'],
        n_batch_fields=options['n_batch_fields'],
        aggregation=keras_aggregate
    )
    training_sequence = build_sequence(
        base_file_names=training_file_names,
        transformation=partial(
            keras_transform,
            size=224,
            augmentation=albumentations.Compose([
                albumentations.HorizontalFlip(),
                albumentations.VerticalFlip(),
                albumentations.RandomScale(.3),
                albumentations.CenterCrop(size, size)
            ])
        )
    )
    validation_sequence = build_sequence(
        base_file_names=validation_file_names,
        transformation=partial(
            keras_transform,
            size=size,
            augmentation=albumentations.Compose([])
        )
    )

    backbone = ResNet18(input_shape=(size, size, 8), include_top=False, weights='imagenet')
    backbone = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=backbone.get_layer('relu1').output
    )
    model = tf.keras.models.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC()
        ]
    )
    history = model.fit(
        training_sequence,
        steps_per_epoch=options['n_training_batches'],
        epochs=100,
        validation_data=validation_sequence,
        validation_steps=options['n_validation_batches'],
        use_multiprocessing=True,
        workers=N_PROCESSES,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(out_path, 'weights.{epoch:02d}.hdf5'),
                verbose=1,
                save_best_only=False
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=.2,
                verbose=1,
                patience=5  # TODO: ?
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
