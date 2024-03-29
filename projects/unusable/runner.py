import json
import multiprocessing
import os
import sys
import time
from argparse import ArgumentParser
from types import SimpleNamespace

from runners.predict_catboost import run


class Namespace(SimpleNamespace):
    def update(self, **kwargs):
        self.__dict__.update(kwargs)


def load_proj():
    if getattr(sys, 'frozen', False):  # if we are inside .exe
        os.environ['PROJ_LIB'] = os.path.join(os.path.split(sys.executable)[0], 'osgeo', 'data', 'proj')
    elif sys.platform == 'win32':
        os.environ['PROJ_LIB'] = os.path.join(os.path.split(sys.executable)[0], '..',
                                              'Lib', 'site-packages', 'osgeo', 'data', 'proj')


if __name__ == '__main__':
    load_proj()

    multiprocessing.freeze_support()
    params_ = Namespace()
    params_.resolution = 30.
    params_.n_processes = 16

    parser = ArgumentParser()
    parser.add_argument('--json', type=str, default='soil_line_unusable.json')
    options = vars(parser.parse_args())
    json_path = options['json']

    override = json.load(open(json_path))
    params_.update(**override)

    print(f'\nloaded params:\n{params_.__dict__}\n')

    t = time.time()
    result = run(**params_.__dict__)
    result = result.rename({s: s + '_rgb' for s in result.index}, axis=0)
    result.to_csv(os.path.join(os.path.dirname(params_.shape_path), 'result.csv'))
    print(f'time={time.time() - t}')
