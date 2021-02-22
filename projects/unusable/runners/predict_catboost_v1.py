import os
import pandas
import geopandas
from argparse import ArgumentParser
from catboost import CatBoostClassifier, Pool

from dataset import BaseDataModule
from utils_v1 import generate_or_read_labels
from catboost_model import catboost_transform, batch_to_numpy


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image-path', type=str, default='/data/unusable/CH/173')
    parser.add_argument('--shape-path', type=str, default='/data/unusable/fields.shp')
    parser.add_argument('--model-path', type=str, default='/data/logs/unusable/model.cbm')  # todo: ...
    parser.add_argument('--resolution', type=float, default=30.)
    parser.add_argument('--n_processes', type=int, default=4)  # todo: 16
    options = parser.parse_args()

    classifier = CatBoostClassifier()
    classifier.load_model(options.model_path)

    fields = geopandas.read_file(options.shape_path).set_index('name')
    labels = generate_or_read_labels(
        image_path=options.image_path,
        fields=fields,
        # todo: remove label_path
        label_path=os.path.join(os.path.dirname(options.shape_path), 'validation.csv')
    )
    result = []
    dataloader = BaseDataModule(
        fields=fields,
        resolution=options.resolution,
        test_labels=labels.iloc[:10],  # todo: ?
        test_image_path=options.image_path,
        n_processes=options.n_processes,
        image_size=None,
        test_transform=catboost_transform,
        batch_size=1
    ).test_dataloader()
    for batch in dataloader:
        result.append(batch_to_numpy(batch))
    data_frame = pandas.DataFrame.from_records(result)
    probabilities = classifier.predict_proba(
        Pool(
            data_frame.drop(['label', 'field_name', 'base_file_name'], axis=1),
            cat_features=['satellite'])
    )[1]
    result = pandas.DataFrame(.0, index=labels.index, columns=labels.columns)
    for probability, field_name, base_file_name in zip(
        probabilities, data_frame['field_name'], data_frame['base_file_name']
    ):
        result.loc[base_file_name, field_name] = probability
    result.to_csv(os.path.join(os.path.dirname(options.model_path), 'result.csv'))
