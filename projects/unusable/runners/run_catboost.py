import os
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from datetime import datetime
from argparse import ArgumentParser
from classification_models.tfkeras import Classifiers

from utils import N_PROCESSES, read_fields, list_tif_files
from sequence import TrainingSequence, concatenate
from transforms import catboost_transform


def run(sequence, n_batches, out_path, n_processes=N_PROCESSES):
    enqueuer = tf.keras.utils.SequenceEnqueuer(sequence, True)
    enqueuer.start(workers=n_processes)
    generator = enqueuer.get()
    results = []
    for i in range(n_batches):
        results.append(generator.next())
    enqueuer.close()
    pd.DataFrame(concatenate(results)).to_csv(out_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/volume/unusable')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable')
    parser.add_argument('--n_training_batches', type=int, default=8)
    parser.add_argument('--n_validation_batches', type=int, default=8)
    parser.add_argument('--n_images', type=int, default=8)
    parser.add_argument('--n_fields', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=224)
    options = vars(parser.parse_args())

    size = options['image_size']
    in_path = options['in_path']
    out_path = os.path.join(options['out_path'], datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(out_path, exist_ok=True)
    tif_path = os.path.join(in_path, 'CH')
    training_file_names = list_tif_files(tif_path, '_174')
    validation_file_names = list_tif_files(tif_path, '_173')
    assert len(training_file_names) + len(validation_file_names) == len(list_tif_files(tif_path, ''))

    fields = read_masks(os.path.join(in_path, 'fields.shp'))
    data_frame = pd.read_excel(os.path.join(in_path, 'NDVI_tif.xls'))
    build_sequence = partial(
        TrainingSequence,
        tif_path=tif_path,
        fields=fields,
        data_frame=data_frame,
        n_batch_images=options['n_batch_images'],
        n_batch_fields=options['n_batch_fields'],
        transform_lambda=catboost_transform
    )
    run(
        sequence=build_sequence(base_file_name=training_file_names),
        n_batches=options['n_training_batches'],
        out_path=os.path.join(out_path, 'training.csv')
    )
    run(
        sequence=build_sequence(base_file_name=validation_file_names),
        n_batches=options['n_validation_batches'],
        out_path=os.path.join(out_path, 'validation.csv')
    )
