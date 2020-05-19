import os
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from datetime import datetime
from argparse import ArgumentParser
from catboost import CatBoostClassifier, Pool

from utils import N_PROCESSES, read_fields, list_tif_files
from sequence import TestSequence, aggregate
from transforms import catboost_transform


def run(sequence, excel_file, out_path, n_processes=N_PROCESSES):
    enqueuer = tf.keras.utils.OrderedEnqueuer(sequence, True)
    enqueuer.start(workers=n_processes)
    generator = enqueuer.get()
    results = []
    for i in range(len(sequence)):
        results.append(next(generator))
        print(f'{i}/{len(sequence)}')
    enqueuer.stop()
    data_frame = pd.DataFrame(aggregate(results, np.concatenate))
    true = {i[1]['NDVI_map'] + i[1]['name'] for i in excel_file.iterrows()}
    data_frame['label'] = [i[1]['file_name'] + i[1]['field_name'] in true for i in data_frame.iterrows()]
    data_frame.to_csv(out_path, index=False)
    print(data_frame['label'].value_counts())
    return Pool(
        data_frame.drop(['label', 'file_name', 'field_name'], axis=1),
        data_frame['label'].astype(np.int),
        cat_features=['satellite']
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/volume/soil_line/unusable')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable')
    parser.add_argument('--n_batch-fields', type=int, default=128)
    parser.add_argument('--image-size', type=int, default=224)
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
    excel_file = pd.read_excel(os.path.join(in_path, 'NDVI_list.xls'))
    excel_file['NDVI_map'] = excel_file['NDVI_map'].apply(lambda x: x[:-6])
    build_sequence = partial(
        TestSequence,
        tif_path=tif_path,
        fields=fields,
        n_batch_fields=options['n_batch_fields'],
        transformation=partial(
            catboost_transform,
            size=size
        ),
        aggregation=partial(
            aggregate,
            aggregation=np.stack
        )
    )
    training_pool = run(
        sequence=build_sequence(base_file_names=training_file_names),
        excel_file=excel_file,
        out_path=os.path.join(out_path, 'training.csv')
    )
    validation_pool = run(
        sequence=build_sequence(base_file_names=validation_file_names),
        excel_file=excel_file,
        out_path=os.path.join(out_path, 'validation.csv')
    )

    classifier = CatBoostClassifier(
        eval_metric='AUC',
        iterations=400,
        learning_rate=.01,
        class_weights=(.01, .99),
        bagging_temperature=10.,
        bootstrap_type='Bayesian'
    )
    classifier.fit(
        training_pool,
        eval_set=validation_pool,
        logging_level='Verbose',
        use_best_model=True
    )
    for metric_name, values in classifier.eval_metrics(
        data=validation_pool,
        metrics=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
    ).items():
        print(metric_name, values[-1])
    for i in np.argsort(classifier.feature_importances_)[::-1]:
        print(classifier.feature_names_[i], classifier.feature_importances_[i])
    classifier.save_model(os.path.join(out_path, 'model.cbm'))
