import os

import gdal
import numpy as np

from config import THRESHOLD, MASKS_PATH, ORIGINAL_PATH, OUTPUT_PATH, ENABLE_INTERACTIVE_MODE, NORMALIZATIONS, PLOTS, \
    MIN_FOR_ELLIPSE
from interactive_mode import interactive_mode
from sc_utils import SATELLITE_CHANNELS, get_stats, get_norm_coefs, linear_transform, save_file, get_region_id, \
    Namespace


class ResultData:
    def __init__(self):
        self.sum_red = None
        self.sum_nir = None
        self.sum_red2 = None
        self.sum_nir2 = None
        self.sum_red_nir = None
        self.num = None
        self.axywh = None

    def process_scene(self, red, nir, mask):
        red2 = red ** 2
        nir2 = nir ** 2
        red_nir = red * nir
        if self.num is None:
            self.sum_red = red
            self.sum_nir = nir
            self.sum_red2 = red2
            self.sum_nir2 = nir2
            self.sum_red_nir = red_nir
            self.num = mask.astype(int)
        else:
            self.sum_red += red
            self.sum_nir += nir
            self.sum_red2 += red2
            self.sum_nir2 += nir2
            self.sum_red_nir += red_nir
            self.num += mask.astype(int)

    def calc_coefficients(self, min_for_data:float):
        with np.errstate(all='ignore'):
            red = self.sum_red / self.num
            nir = self.sum_nir / self.num
            red2 = np.nan_to_num(self.sum_red2 / self.num) - red ** 2
            nir2 = np.nan_to_num(self.sum_nir2 / self.num) - nir ** 2
            red_nir = np.nan_to_num(self.sum_red_nir / self.num) - red * nir

            self.axywh = np.zeros(list(red.shape) + [5])

            alpha = (red2 - nir2 + ((red2 - nir2) ** 2 + 4 * red_nir ** 2) ** .5) / red_nir / 2
            w = (red2 + red_nir / alpha) ** .5
            h = (red2 - red_nir * alpha) ** .5
            self.axywh[:, :, :, 0] = 1 / alpha
            self.axywh[:, :, :, 1] = red
            self.axywh[:, :, :, 2] = nir
            self.axywh[:, :, :, 3] = w
            self.axywh[:, :, :, 4] = h

            self.axywh = np.nan_to_num(self.axywh) * (self.num >= min_for_data)[..., np.newaxis]


class Scene:
    def __init__(self, scene_name, params):
        sat = SATELLITE_CHANNELS[[s for s in SATELLITE_CHANNELS.keys() if s in scene_name][0]]
        sat = {v: k for k, v in sat.items()}
        self.scene_name = scene_name
        self.params = params
        self.red_path = os.path.join(params.ORIGINAL_PATH, scene_name + f'_red_{sat["red"]}.tif')
        self.nir_path = os.path.join(params.ORIGINAL_PATH, scene_name + f'_nir_{sat["nir"]}.tif')
        self.mask_path = os.path.join(params.MASKS_PATH, scene_name + '.tif')
        ds = gdal.Open(self.red_path)
        if ds is None:
            raise FileNotFoundError
        self.geotransform = ds.GetGeoTransform()
        self.projection = ds.GetProjection()

    def process(self, normalizations=None):
        ds = gdal.Open(self.red_path)
        if ds is None:
            return
        band = ds.GetRasterBand(1)
        red = band.ReadAsArray().astype(float)
        ds = gdal.Open(self.nir_path)
        if ds is None:
            return
        band = ds.GetRasterBand(1)
        nir = band.ReadAsArray().astype(float)
        ds = gdal.Open(self.mask_path)
        if ds is None:
            return
        band = ds.GetRasterBand(1)
        mask = band.ReadAsArray().astype(float)
        mask = mask > self.params.THRESHOLD

        red_stats = get_stats(red, mask)
        nir_stats = get_stats(nir, mask)

        norm_coefs = get_norm_coefs(red_stats, nir_stats, normalizations)

        normalized = [linear_transform(red, nir, mask, coefs) for coefs in norm_coefs]

        r, n, m = np.array([q[0] for q in normalized]), \
                  np.array([q[1] for q in normalized]), \
                  np.array([q[2] for q in normalized])

        return r, n, m


def run(params):
    scene_names = os.listdir(params.MASKS_PATH)
    scene_names = [s[:-4] for s in scene_names]

    result_data = ResultData()
    scenes = []

    region_ids = set(get_region_id(s) for s in scene_names)
    region_id = list(region_ids)[0]  # todo

    for s in scene_names:
        if region_id in s:
            print(s)
            scenes.append(Scene(s, params))
            r, n, m = scenes[-1].process(params.NORMALIZATIONS)
            result_data.process_scene(r, n, m)

    result_data.calc_coefficients(min_for_data=params.MIN_FOR_ELLIPSE)

    if params.ENABLE_INTERACTIVE_MODE:
        interactive_mode(scenes, result_data, params)

    geotransform = scenes[0].geotransform
    projection = scenes[0].projection

    if not os.path.exists(params.OUTPUT_PATH):
        os.makedirs(params.OUTPUT_PATH)

    for i, norm in enumerate(params.NORMALIZATIONS):
        axywh = result_data.axywh[i]
        a, x, y, w, h =  tuple(axywh.transpose([2, 0, 1]))

        save_file(f'{params.OUTPUT_PATH}/A_norm_{norm}.tif', a, geotransform, projection)
        save_file(f'{params.OUTPUT_PATH}/X_norm_{norm}.tif', x, geotransform, projection)
        save_file(f'{params.OUTPUT_PATH}/Y_norm_{norm}.tif', y, geotransform, projection)
        save_file(f'{params.OUTPUT_PATH}/W_norm_{norm}.tif', w, geotransform, projection)
        save_file(f'{params.OUTPUT_PATH}/H_norm_{norm}.tif', h, geotransform, projection)
        save_file(f'{params.OUTPUT_PATH}/B_norm_{norm}.tif', y - a * x, geotransform, projection)
        save_file(f'{params.OUTPUT_PATH}/C_norm_{norm}.tif', x + y, geotransform, projection)


if __name__ == '__main__':
    params_ = Namespace()
    params_.THRESHOLD = THRESHOLD
    params_.MIN_FOR_ELLIPSE = MIN_FOR_ELLIPSE
    params_.MASKS_PATH = MASKS_PATH
    params_.ORIGINAL_PATH = ORIGINAL_PATH
    params_.OUTPUT_PATH = OUTPUT_PATH
    params_.ENABLE_INTERACTIVE_MODE = ENABLE_INTERACTIVE_MODE
    params_.NORMALIZATIONS = NORMALIZATIONS
    params_.PLOTS = PLOTS

    run(params=params_)
