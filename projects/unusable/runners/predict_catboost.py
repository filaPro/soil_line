import os
import pandas as pd
import tensorflow as tf
from functools import partial
from argparse import ArgumentParser
from catboost import CatBoostClassifier, Pool

from utils import N_PROCESSES, read_fields, list_tif_files
from sequence import TestSequence
from transforms import catboost_transform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/volume/unusable')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable/2020-03-20-18-37-05')
    parser.add_argument('--n_batch_fields', type=int, default=128)
    parser.add_argument('--image-size', type=int, default=224)
    options = vars(parser.parse_args())

    size = options['image_size']
    in_path = options['in_path']
    out_path = options['out_path']
    tif_path = os.path.join(in_path, 'CH')
    base_file_names = list_tif_files(tif_path, '_173')

    classifier = CatBoostClassifier()
    classifier.load_model(os.path.join(out_path, 'model.cbm'))

    fields = read_fields(os.path.join(in_path, 'fields.shp'))
    result = pd.DataFrame(.0, index=base_file_names, columns=list(fields.keys()))
    sequence = TestSequence(
        tif_path=tif_path,
        base_file_names=base_file_names,
        fields=fields,
        n_batch_fields=options['n_batch_fields'],
        transform_lambda=partial(
            catboost_transform,
            size=size
        )
    )
    enqueuer = tf.keras.utils.OrderedEnqueuer(sequence, True)
    enqueuer.start(workers=N_PROCESSES)
    generator = enqueuer.get()
    for i in range(len(sequence)):
        print(f'{i}/{len(sequence)}')
        data_frame = pd.DataFrame(next(generator))
        probabilities = classifier.predict_proba(Pool(
            data_frame.drop(['label', 'file_name', 'field_name'], axis=1),
            cat_features=['satellite']
        ))[:, 1]
        for j in range(len(data_frame)):
            result.loc[data_frame['file_name'][j], data_frame['field_name'][j]] = probabilities[j]
    enqueuer.stop()
    result.to_csv(os.path.join(out_path, 'result.csv'))
