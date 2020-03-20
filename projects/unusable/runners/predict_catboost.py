import os
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from datetime import datetime
from argparse import ArgumentParser
from catboost import CatBoostClassifier

from utils import N_PROCESSES, read_fields, list_tif_files
from sequence import TestSequence, concatenate
from transforms import catboost_transform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable/TODO')
    parser.add_argument('--n_batch_fields', type=int, default=4)
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
    result = pd.Dataframe(.0, index=base_file_names, columns=list(fields.keys()))
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
    enqueuer.start(workers=n_processes)
    generator = enqueuer.get()
    for batch in generator:
        if len(batch) > 0:
            data_frame = pd.DataFrame(batch)
            probabilities = classifier.predict_proba(data_frame)
            for i in range(len(data_frame)):
                result.loc[data_frame['file_name'][i], data_frame['field_name'][i]] = probabilities[i]
    enqueuer.stop()
    results.to_csv(os.path.join(out_path, 'result.csv'))

