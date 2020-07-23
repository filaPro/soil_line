from types import SimpleNamespace

import gdal
import numpy as np

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
        '12': 'swir2'
    }
}


def get_region_id(s: str):
    return s.split('_')[-1]


def get_stats(arr, mask):
    q = arr[mask]
    if not len(q):
        return {
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'q05': np.nan,
            'q01': np.nan,
            'q00': np.nan,
        }
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
        'q00': q09
    }


def get_norm_coefs(red_stats, nir_stats, normalizations=None):
    if normalizations is None:
        normalizations = ['none']
    res = []
    for norm in normalizations:
        if norm == 'disp':
            res.append([[red_stats['mean'], red_stats['std']],
                        [nir_stats['mean'], nir_stats['std']]])
        elif norm == 'none':
            res.append([[0, 1], [0, 1]])
        else:
            raise ValueError(f'Unknown normalization {norm}')
    return res


def linear_transform(red, nir, mask, coefs):
    r = np.nan_to_num((red - coefs[0][0]) / coefs[0][1]) * mask
    n = np.nan_to_num((nir - coefs[1][0]) / coefs[1][1]) * mask
    m = mask
    return r, n, m

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
