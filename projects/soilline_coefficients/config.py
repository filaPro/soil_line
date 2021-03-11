import tempfile
import os


ORIGINAL_PATH = 'D:/DATA/soilline_coefficients/OUT1_CH'
MASKS_PATH = 'D:/DATA/soilline_coefficients/OUT1_masks'
OUTPUT_PATH = os.path.join(tempfile.gettempdir(), 'out')

THRESHOLD = .5
MIN_FOR_ELLIPSE = 6

ENABLE_INTERACTIVE_MODE = True

NORMALIZATIONS = ['none']

PLOTS = os.path.join(tempfile.gettempdir(), 'ell.png')  # possible values are also 'show' and 'none'
REPROJECT_PATH = os.path.join(tempfile.gettempdir(), 'reproject')
