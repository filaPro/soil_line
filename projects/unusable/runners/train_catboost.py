import os
import pandas
import logging
import geopandas
import numpy as np
from functools import partial
from argparse import ArgumentParser
from catboost import CatBoostClassifier, Pool

from dataset import BaseDataModule
from utils_v1 import generate_or_read_labels
from catboost_model import catboost_transform, batch_to_numpy


def generate_or_read_pool(fields, resolution, labels, image_path, n_processes, out_path=None):
    # do nothing if pool.scv already exists
    if out_path is not None and os.path.exists(out_path):
        data_frame = pandas.read_csv(out_path)
    else:
        result = []
        dataloader = BaseDataModule(
            fields=fields,
            resolution=resolution,
            test_labels=labels,
            test_image_path=image_path,
            n_processes=n_processes,
            image_size=None,
            test_transform=catboost_transform,
            batch_size=1,
        ).test_dataloader()
        for batch in dataloader:
            result.append(batch_to_numpy(batch))
        data_frame = pandas.DataFrame.from_records(result)

    if out_path is not None:
        data_frame.to_csv(out_path, index=False)
    return Pool(
        data_frame.drop(['label', 'field_name', 'base_file_name'], axis=1),
        data_frame['label'],
        cat_features=['satellite']
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training-image-path', type=str, default='/data/soil_line/unusable/CH/174')
    parser.add_argument('--validation-image-path', type=str, default='/data/soil_line/unusable/CH/173')
    parser.add_argument('--shape-path', type=str, default='/data/soil_line/unusable/fields.shp')
    parser.add_argument('--excel-path', type=str, default='/data/soil_line/unusable/NDVI_list.xls')
    parser.add_argument('--log-path', type=str, default='/data/logs/unusable/...')
    parser.add_argument('--resolution', type=float, default=30.)
    parser.add_argument('--n_processes', type=int, default=16)
    options = parser.parse_args()

    fields = geopandas.read_file(options.shape_path).set_index('name')
    label_lambda = partial(
        generate_or_read_labels,
        excel_path=options.excel_path,
        fields=fields
    )
    training_labels = label_lambda(
        image_path=options.training_image_path,
        label_path=os.path.join(os.path.dirname(options.excel_path), 'training.csv')
    )
    validation_labels = label_lambda(
        image_path=options.validation_image_path,
        label_path=os.path.join(os.path.dirname(options.excel_path), 'validation.csv')
    )
    pool_lambda = partial(
        generate_or_read_pool,
        fields=fields,
        resolution=options.resolution,
        n_processes=options.n_processes
    )
    training_pool = pool_lambda(
        labels=training_labels,
        image_path=options.training_image_path,
        out_path=os.path.join(options.log_path, 'training.csv'),
    )
    validation_pool = pool_lambda(
        labels=validation_labels,
        image_path=options.validation_image_path,
        out_path=os.path.join(options.log_path, 'validation.csv'),
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
        logging.info(f'{metric_name}: {values[-1]}')
    for i in np.argsort(classifier.feature_importances_)[::-1]:
        logging.info(f'{classifier.feature_names_[i]}: '
                     f'{classifier.feature_importances_[i]}')
    classifier.save_model(os.path.join(options.log_path, 'model.cbm'))
