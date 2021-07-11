import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

from predict import run


def load_proj():
    if getattr(sys, 'frozen', False):  # if we are inside .exe
        os.environ['PROJ_LIB'] = os.path.join(os.path.split(sys.executable)[0], 'osgeo', 'data', 'proj')
    elif sys.platform == 'win32':
        os.environ['PROJ_LIB'] = os.path.join(os.path.split(sys.executable)[0], '..',
                                              'Lib', 'site-packages', 'osgeo', 'data', 'proj')


class Namespace(SimpleNamespace):
    def update(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == '__main__':
    load_proj()

    params = Namespace()
    params.batch_size = 32
    params.size = 512
    params.step = 256
    params.quantile = .05

    parser = ArgumentParser()
    parser.add_argument('--json', type=str, default='soil_line_bare.json')
    options = vars(parser.parse_args())
    json_path = options['json']

    override = json.load(open(json_path))
    params.update(**override)
    size = params.size

    run(**params.__dict__)
