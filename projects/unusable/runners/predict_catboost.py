import os
from argparse import ArgumentParser

import geopandas
import pandas
from catboost import CatBoostClassifier, Pool

from catboost_model import catboost_transform, batch_to_numpy
from dataset import BaseDataModule
from utils import generate_or_read_labels


def run(image_path, shape_path, model_path, resolution, n_processes):
    classifier = CatBoostClassifier()
    classifier.load_model(model_path)

    fields = geopandas.read_file(shape_path).set_index('name')
    labels = generate_or_read_labels(
        image_path=image_path,
        fields=fields,
        # uncomment for faster debug
        label_path=os.path.join(os.path.dirname(options.shape_path), 'validation.csv')
    )
    result = []
    dataloader = BaseDataModule(
        fields=fields,
        resolution=resolution,
        test_labels=labels,
        test_image_path=image_path,
        n_processes=n_processes,
        image_size=None,
        test_transform=catboost_transform,
        batch_size=1
    ).test_dataloader()
    for batch in dataloader:
        result.append(batch_to_numpy(batch))
    data_frame = pandas.DataFrame.from_records(result)
    # uncomment for faster debug and comment previous lines
    # data_frame = pandas.read_csv(os.path.join(os.path.dirname(model_path), 'validation.csv'))

    probabilities = classifier.predict_proba(Pool(
        data_frame.drop(['label', 'field_name', 'base_file_name'], axis=1),
        cat_features=['satellite']
    ))[:, 1]
    result = pandas.DataFrame(.0, index=labels.index, columns=labels.columns)
    for probability, field_name, base_file_name in zip(
        probabilities, data_frame['field_name'], data_frame['base_file_name']
    ):
        result.loc[base_file_name, field_name] = probability
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image-path', type=str, default='/data/soil_line/unusable/CH/173')
    parser.add_argument('--shape-path', type=str, default='/data/soil_line/unusable/fields_v2/fields.shp')
    parser.add_argument('--model-path', type=str, default='/logs/unusable/.../model.cbm')
    parser.add_argument('--resolution', type=float, default=30.)
    parser.add_argument('--n-processes', type=int, default=16)
    options = parser.parse_args()

    result = run(
        image_path=options.image_path,
        shape_path=options.shape_path,
        model_path=options.model_path,
        resolution=options.resolution,
        n_processes=options.n_processes
    )

    result_path = options.model_path + '_results/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result.to_csv(os.path.join(result_path, 'result.csv'))
