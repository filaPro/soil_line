import logging
import os
import sys
from types import SimpleNamespace

import gdal
import numpy as np

logger = logging.Logger('logger', level=0)


def get_stream_handler():
    _log_format = f"%(asctime)s - [%(levelname)s] - (%(filename)s)(%(lineno)d) - %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(0)
    stream_handler.setFormatter(logging.Formatter(_log_format))
    return stream_handler


logger.addHandler(get_stream_handler())

SATELLITE_CHANNELS = {
    'LT04': {
        '01': 'blue',
        '02': 'green',
        '03': 'red',
        '04': 'nir',
        '05': 'swir1',
        '07': 'swir2'
    },
    'LT05': {
        '01': 'blue',
        '02': 'green',
        '03': 'red',
        '04': 'nir',
        '05': 'swir1',
        '07': 'swir2'
    },
    'LE07': {
        '01': 'blue',
        '02': 'green',
        '03': 'red',
        '04': 'nir',
        '05': 'swir1',
        '07': 'swir2'
    },
    'LC08': {
        '02': 'blue',
        '03': 'green',
        '04': 'red',
        '05': 'nir',
        '06': 'swir1',
        '07': 'swir2'
    },
    'S2AB': {
        '02': 'blue',
        '03': 'green',
        '04': 'red',
        '08': 'nir',
        '11': 'swir1',
        '12': 'swir2'
    }
}


def get_region_id(s: str):
    return s.split('_')[-1]


def get_stats(arr, mask):
    try:
        q = arr[mask]
        l, r = np.quantile(q, .03), np.quantile(q, .97)
        q = q[q < r]
        q = q[q > l]
        qmin = q.min()
        qmax = q.max()
        qmean = q.mean()
        q05 = np.quantile(q, 0.5)
        q01 = np.quantile(q, 0.1)
        q09 = np.quantile(q, 0.9)
        qstd = np.std(q)
        return {
            'min': qmin,
            'max': qmax,
            'mean': qmean,
            'std': qstd,
            'q05': q05,
            'q01': q01,
            'q09': q09
        }
    except Exception as e:
        logger.warning(f'Exception in scene processing: {e}')
        return {
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'q05': np.nan,
            'q01': np.nan,
            'q09': np.nan,
        }


def get_norm_coefs(stats, normalizations=None):
    if normalizations is None:
        normalizations = ['none']
    res = []
    for norm in normalizations:
        if norm == 'disp':
            res.append({key: [stats[key]['mean'], stats[key]['std']] for key in stats})
        elif norm == 'none':
            res.append({key: [0, 1] for key in stats})
        else:
            raise ValueError(f'Unknown normalization {norm}')
    return res


def linear_transform(channel_value, mask, coefs):
    return {key: np.nan_to_num((channel_value[key] - coefs[key][0]) / coefs[key][1]) * mask for key in channel_value}


def get_align_coefficients(g1, g2, shape1, shape2):
    if (g1[1] / g2[1] - 1) ** 2 + (g1[5] / g2[5] - 1) ** 2 > .001:
        raise NotImplementedError
    x, y = int((g2[3] - g1[3]) / g1[5]), int((g2[0] - g1[0]) / g1[1])
    dx, dy = 1, 1
    UP1, DOWN1, LEFT1, RIGHT1 = (max(-x, 0), max(x + shape2[0] - shape1[0], 0),
                                 max(-y, 0), max(y + shape2[1] - shape1[1], 0))

    UP2, DOWN2, LEFT2, RIGHT2 = (max(x, 0), max(-x - shape2[0] + shape1[0], 0),
                                 max(y, 0), max(-y - shape2[1] + shape1[1], 0))

    g = list(g1)
    g[0] -= LEFT1 * g[1]
    g[3] -= UP1 * g[5]
    g = tuple(g)
    return x, y, dx, dy, (UP1, DOWN1, LEFT1, RIGHT1), (UP2, DOWN2, LEFT2, RIGHT2), g


def pad_udlr(arr, udlr):
    UP1, DOWN1, LEFT1, RIGHT1 = udlr
    if len(arr.shape) == 3:
        pad_dims = ((0, 0), (UP1, DOWN1), (LEFT1, RIGHT1))
    elif len(arr.shape) == 2:
        pad_dims = ((UP1, DOWN1), (LEFT1, RIGHT1))
    else:
        raise ValueError('Image dimension should be 2 or 3.')
    return np.pad(arr, pad_dims)


def align_images(a: tuple, b: tuple, a_gt, b_gt):
    x, y, dx, dy, udlr1, udlr2, gt = get_align_coefficients(a_gt, b_gt, a[0].shape[-2:], b[0].shape[-2:])
    if not dx == dy == 1:
        raise NotImplementedError('Image has unexpected pixel size.')

    if sum(udlr1) + sum(udlr2):
        logger.info(f'Align images:{udlr1, udlr2}')
        a = tuple(pad_udlr(q, udlr1) for q in a)
        b = tuple(pad_udlr(q, udlr2) for q in b)

    return a, b, gt


def save_file(fn, arr, geotransform, projection):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(fn, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(geotransform)
    outdata.SetProjection(projection)
    outdata.GetRasterBand(1).WriteArray(arr)
    outdata.FlushCache()  # saves to disk


class Namespace(SimpleNamespace):
    def update(self, **kwargs):
        self.__dict__.update(kwargs)


def load_proj():
    if getattr(sys, 'frozen', False):  # if we are inside .exe
        # noinspection PyUnresolvedReferences, PyProtectedMember
        os.environ['PROJ_LIB'] = os.path.join(sys._MEIPASS, 'proj')
    # elif sys.platform == 'win32':
    #     os.environ['PROJ_LIB'] = os.path.join(os.path.split(sys.executable)[0], 'Library', 'share', 'proj')
