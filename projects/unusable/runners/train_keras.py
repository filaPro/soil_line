import os
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import albumentations
from functools import partial
from argparse import ArgumentParser

from utils import N_PROCESSES, read_fields, list_tif_files
from sequence import TrainingSequence, keras_aggregate
from transforms import keras_transform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training-image-path', type=str, default='/volume/soil_line/unusable/CH/174')
    parser.add_argument('--validation-image-path', type=str, default='/volume/soil_line/unusable/CH/173')
    parser.add_argument('--shape-path', type=str, default='/volume/soil_line/unusable/fields.shp')
    parser.add_argument('--excel-path', type=str, default='/volume/soil_line/unusable/NDVI_list.xls')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable/...')
    parser.add_argument('--n-training-batches', type=int, default=500)
    parser.add_argument('--n-validation-batches', type=int, default=100)
    parser.add_argument('--n-batch-images', type=int, default=64)
    parser.add_argument('--n-batch-fields', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=128)
    options = parser.parse_args()

    os.makedirs(options.out_path, exist_ok=True)
    fields, spatial_reference = read_fields(options.shape_path)
    data_frame = pd.read_excel(options.excel_path)
    build_sequence = partial(
        TrainingSequence,
        fields=fields,
        spatial_reference=spatial_reference,
        data_frame=data_frame,
        n_batch_images=options.n_batch_images,
        n_batch_fields=options.n_batch_fields,
        aggregation=keras_aggregate
    )
    training_sequence = build_sequence(
        tif_path=options.training_image_path,
        base_file_names=list_tif_files(options.training_image_path),
        transformation=partial(
            keras_transform,
            size=options.image_size,
            augmentation=albumentations.Compose([
                albumentations.HorizontalFlip(),
                albumentations.VerticalFlip()
            ])
        )
    )
    validation_sequence = build_sequence(
        tif_path=options.validation_image_path,
        base_file_names=list_tif_files(options.validation_image_path),
        transformation=partial(
            keras_transform,
            size=options.image_size,
            augmentation=albumentations.Compose([])
        )
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(None, None, 8)),
        tf.keras.layers.Conv2D(64, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(128, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(256, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(512, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(512, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(512, activation='relu'),
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
        steps_per_epoch=options.n_training_batches,
        epochs=100,
        validation_data=validation_sequence,
        validation_steps=options.n_validation_batches,
        use_multiprocessing=True,
        workers=N_PROCESSES,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(options.out_path, 'weights.{epoch:02d}.hdf5'),
                verbose=1,
                save_best_only=False
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=.2,
                verbose=1,
                patience=3
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=options.out_path,
                write_graph=False,
                profile_batch=0
            ),
            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(options.out_path, 'log.csv'),
                append=True
            )
        ]
    )
