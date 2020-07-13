import os
import numpy as np
import pandas as pd
import albumentations
import tensorflow as tf
from functools import partial
from argparse import ArgumentParser

from utils import N_PROCESSES, read_fields, list_tif_files
from sequence import TestSequence, aggregate
from transforms import keras_transform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/volume/soil_line/unusable')
    parser.add_argument('--model-path', type=str, default='/volume/logs/unusable/.../....hdf5')
    parser.add_argument('--n-batch-fields', type=int, default=256)
    parser.add_argument('--image-size', type=int, default=128)
    options = vars(parser.parse_args())

    size = options['image_size']
    in_path = options['in_path']
    model_path = options['model_path']
    tif_path = os.path.join(in_path, 'CH')
    base_file_names = list_tif_files(tif_path, '_173')

    model = tf.keras.models.load_model(model_path)
    fields, spatial_reference = read_fields(os.path.join(in_path, 'fields.shp'))
    result = pd.DataFrame(.0, index=base_file_names, columns=list(fields.keys()))
    sequence = TestSequence(
        tif_path=tif_path,
        base_file_names=base_file_names,
        fields=fields,
        spatial_reference=spatial_reference,
        n_batch_fields=options['n_batch_fields'],
        transformation=partial(
            keras_transform,
            size=size,
            augmentation=albumentations.Compose([])
        ),
        aggregation=partial(
            aggregate,
            aggregation=np.stack
        )
    )
    enqueuer = tf.keras.utils.OrderedEnqueuer(sequence, True)
    enqueuer.start(workers=N_PROCESSES)
    generator = enqueuer.get()
    for i in range(len(sequence)):
        print(f'{i}/{len(sequence)}')
        batch = next(generator)
        probabilities = model.predict_on_batch(batch['image'])
        for j in range(len(probabilities)):
            result.loc[batch['file_name'][j], batch['field_name'][j]] = probabilities[j]
    enqueuer.stop()
    result.to_csv(os.path.join(os.path.dirname(model_path), 'result.csv'))
