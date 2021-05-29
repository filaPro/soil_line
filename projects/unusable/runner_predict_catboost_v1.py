import json
import multiprocessing
import os
import sys
import time
from argparse import ArgumentParser
from types import SimpleNamespace

from runners.predict_catboost_v1 import run


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
    parser.add_argument('--json', type=str, default='soilline_unusable_config.json')
    options = vars(parser.parse_args())
    json_path = options['json']

    override = json.load(open(json_path))
    params_.update(**override)

    print(f'\nloaded params:\n{params_.__dict__}\n')

    t = time.time()
    run(**params_.__dict__)
    print(f'time={time.time() - t}')
