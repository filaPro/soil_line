import tempfile
import os


ORIGINAL_PATH = '/media/hdd/OUT1/CH'
MASKS_PATH = '/home/fila/data/logs/bare/2020-07-19-13-47-173-and-174/masks'
OUTPUT_PATH = os.path.join(tempfile.gettempdir(), 'out')

THRESHOLD = .5
MIN_FOR_ELLIPSE = 6

ENABLE_INTERACTIVE_MODE = True

NORMALIZATIONS = ['none']

PLOTS = os.path.join(tempfile.gettempdir(), 'ell.png')  # possible values are also 'show' and 'none'
