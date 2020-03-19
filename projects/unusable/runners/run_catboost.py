import os
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from datetime import datetime
from argparse import ArgumentParser
from catboost import CatBoostClassifier

from utils import N_PROCESSES, read_fields, list_tif_files
from sequence import TrainingSequence, concatenate
from transforms import catboost_transform


def run(sequence, n_batches, out_path, n_processes=N_PROCESSES):
    enqueuer = tf.keras.utils.OrderedEnqueuer(sequence, True)
    enqueuer.start(workers=n_processes)
    generator = enqueuer.get()
    results = []
    for i in range(n_batches):
        results.append(next(generator))
        print(f'{i}/{n_batches}')
    enqueuer.stop()
    data_frame = pd.DataFrame(concatenate(results, np.concatenate))
    data_frame.to_csv(out_path, index=False)
    print(data_frame['label'].value_counts())
    return data_frame


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/volume/unusable')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable')
    parser.add_argument('--n_training_batches', type=int, default=1)
    parser.add_argument('--n_validation_batches', type=int, default=1)
    parser.add_argument('--n_batch_images', type=int, default=50)
    parser.add_argument('--n_batch_fields', type=int, default=4)
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

    fields = read_fields(os.path.join(in_path, 'fields.shp'))
    data_frame = pd.read_excel(os.path.join(in_path, 'NDVI_list.xls'))
    build_sequence = partial(
        TrainingSequence,
        tif_path=tif_path,
        fields=fields,
        data_frame=data_frame,
        n_batch_images=options['n_batch_images'],
        n_batch_fields=options['n_batch_fields'],
        transform_lambda=catboost_transform
    )
    training_data_frame = run(
        sequence=build_sequence(base_file_names=training_file_names),
        n_batches=options['n_training_batches'],
        out_path=os.path.join(out_path, 'training.csv')
    )
    validation_data_frame = run(
        sequence=build_sequence(base_file_names=validation_file_names),
        n_batches=options['n_validation_batches'],
        out_path=os.path.join(out_path, 'validation.csv')
    )

    training_data = training_data_frame.drop(['label'], axis=1)
    validation_data = validation_data_frame.drop(['label'], axis=1)
    training_label = training_data_frame['label'].astype(np.int)
    validation_label = validation_data_frame['label'].astype(np.int)
    classifier = CatBoostClassifier(
        eval_metric='AUC'
    )
    classifier.fit(
        training_data,
        training_label,
        eval_set=(validation_data, validation_label),
        cat_features=['satellite'],
        logging_level='Verbose',
        use_best_model=True
    )
    classifier.save_model(os.path.join(out_path, 'model.cbm'))
