import json
import os
import sys
import time
import traceback
from argparse import ArgumentParser

from app import run as run_app
from classify import run as run_classify
from lib import load_proj
from preprocess import run as run_preprocess


def run_preprocess_(**params):
    in_path = params['in_path']
    return run_preprocess(
        tif_path=os.path.join(in_path, 'NDVI_tif'),
        out_path=os.path.join(in_path, 'out', 'tif'),
        **params)


def run_app_(**params):
    in_path = params.pop('in_path')
    return run_app(
        tif_path=os.path.join(in_path, 'NDVI_tif'),
        shape_path=os.path.join(in_path, 'fields.shp'),
        excel_path=os.path.join(in_path, 'NDVI_list.xls'),
        out_path=os.path.join(in_path, 'out', 'deviations'),
        **params)


def run_classify_(**params):
    in_path = params['in_path']
    return run_classify(
        out_path=os.path.join(os.path.dirname(in_path), 'classes'),
        **params)


if __name__ == '__main__':
    try:
        load_proj()

        parser = ArgumentParser()
        parser.add_argument('--json', type=str, default='soil_line_fields.json')

        print(parser.parse_args())

        with open(parser.parse_args().json, 'r') as f:
            meta_args = json.load(f)

        preprocess_params = {
            'in_path': '/volume',
            'fill_method': 'ns',
        }

        app_params = {
            'in_path': '/volume',
            'buffer_size': 0,
            'resolution': 10.0,
            'min_quantile': 0.0,
            'max_quantile': 1.0,
            'fill_method': 'ns',
            'aggregation_method': 'mean',
            'year_aggregation_method': 1,
            'dilation_method': 3,
            'deviation_method': 1
        }

        classify_params = {
            'in_path': '/volume/out/deviations',
            'n_classes': 3,
            'sieve_threshold': 0,
            'method': 's',
            'missing_value': -1.0
        }

        if 'preprocess_params' in meta_args:
            preprocess_params.update(**meta_args['preprocess_params'])
        if 'app_params' in meta_args:
            app_params.update(**meta_args['app_params'])
        if 'classify_params' in meta_args:
            classify_params.update(**meta_args['classify_params'])

        for task in meta_args['tasks']:
            print(f'\n====== Running {task} =======\n')
            if task == 'preprocess':
                run_preprocess_(**preprocess_params)
            elif task == 'app':
                run_app_(**app_params)
            elif task == 'classify':
                run_classify_(**classify_params)
            else:
                raise ValueError(f'Unknown task {task}')
    except Exception as e:
        traceback.print_exception(*sys.exc_info())
        time.sleep(.2)
        print('\n\n', e)

    print('\n Press <Enter> to continue...')
    sys.stdin.readline()
