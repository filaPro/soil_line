import os
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
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
    parser.add_argument('--training-image-path', type=str, default='/volume/soil_line/unusable/CH/174')
    parser.add_argument('--validation-image-path', type=str, default='/volume/soil_line/unusable/CH/173')
    parser.add_argument('--shape-path', type=str, default='/volume/soil_line/unusable/fields.shp')
    parser.add_argument('--excel-path', type=str, default='/volume/soil_line/unusable/NDVI_list.xls')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable/...')
    parser.add_argument('--n_batch-fields', type=int, default=128)
    parser.add_argument('--image-size', type=int, default=224)
    options = parser.parse_args()

    os.makedirs(options.out_path, exist_ok=True)
    fields, spatial_reference = read_fields(options.shape_path)
    excel_file = pd.read_excel(options.excel_path)
    excel_file['NDVI_map'] = excel_file['NDVI_map'].apply(lambda x: x[:-6])
    build_sequence = partial(
        TestSequence,
        fields=fields,
        spatial_reference=spatial_reference,
        n_batch_fields=options.n_batch_fields,
        transformation=partial(
            catboost_transform,
            size=options.image_size
        ),
        aggregation=partial(
            aggregate,
            aggregation=np.stack
        )
    )
    training_pool = run(
        sequence=build_sequence(
            tif_path=options.training_image_path,
            base_file_names=list_tif_files(options.training_image_path)
        ),
        excel_file=excel_file,
        out_path=os.path.join(options.out_path, 'training.csv')
    )
    validation_pool = run(
        sequence=build_sequence(
            tif_path=options.validation_image_path,
            base_file_names=list_tif_files(options.validation_image_path)
        ),
        excel_file=excel_file,
        out_path=os.path.join(options.out_path, 'validation.csv')
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
    classifier.save_model(os.path.join(options.out_path, 'model.cbm'))
