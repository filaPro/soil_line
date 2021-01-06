import json
import os
import sys
import time
import traceback
from argparse import ArgumentParser

import gdal
import numpy as np

from config import THRESHOLD, MASKS_PATH, ORIGINAL_PATH, OUTPUT_PATH, ENABLE_INTERACTIVE_MODE, NORMALIZATIONS, PLOTS, \
    MIN_FOR_ELLIPSE, REPROJECT_PATH
from interactive_mode import interactive_mode
from reproject import open_with_reproject
from sc_utils import SATELLITE_CHANNELS, get_stats, get_norm_coefs, linear_transform, save_file, \
    Namespace, load_proj, get_transform_coefficients


class ResultData:
    def __init__(self):
        self.sum_red = None
        self.sum_nir = None
        self.sum_red2 = None
        self.sum_nir2 = None
        self.sum_red_nir = None
        self.num = None
        self.axywh = None

        self.geo_transform = None

    def process_scene(self, red, nir, mask, geo_transform):
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
            self.geo_transform = geo_transform
        else:
            if True:  # list(geo_transform) != list(self.geo_transform):
                x, y, dx, dy = get_transform_coefficients(self.geo_transform, geo_transform)
                if not dx == dy == 1:
                    print('Image has unexpected pixel size.')
                    return

                UP1, DOWN1, LEFT1, RIGHT1 = (max(-x, 0), max(x + red.shape[1] - self.sum_red.shape[1], 0),
                                             max(-y, 0), max(y + red.shape[2] - self.sum_red.shape[2], 0))

                UP2, DOWN2, LEFT2, RIGHT2 = (max(x, 0), max(-x - red.shape[1] + self.sum_red.shape[1], 0),
                                             max(y, 0), max(-y - red.shape[2] + self.sum_red.shape[2], 0))

                if UP1+DOWN1+LEFT1+RIGHT1+UP2+DOWN2+LEFT2+RIGHT2:
                    print(UP1, DOWN1, LEFT1, RIGHT1, UP2, DOWN2, LEFT2, RIGHT2)

                (self.sum_red, self.sum_nir, self.sum_red2, self.sum_nir2, self.sum_red_nir, self.num) = \
                    (np.pad(a, ((0, 0), (UP1, DOWN1), (LEFT1, RIGHT1))) for a in
                     (self.sum_red, self.sum_nir, self.sum_red2, self.sum_nir2, self.sum_red_nir, self.num))

                (red, nir, red2, nir2, red_nir, mask) = \
                    (np.pad(a, ((0, 0), (UP2, DOWN2), (LEFT2, RIGHT2))) for a in
                     (red, nir, red2, nir2, red_nir, mask))

                g = list(self.geo_transform)
                g[0] -= DOWN1 * g[1]
                g[3] -= UP1 * g[5]
                self.geo_transform = tuple(g)
            self.sum_red += red
            self.sum_nir += nir
            self.sum_red2 += red2
            self.sum_nir2 += nir2
            self.sum_red_nir += red_nir
            self.num += mask.astype(int)

    def calc_coefficients(self, min_for_data: float):
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

        self.red_ds, self.nir_ds, self.mask_ds = None, None, None

    def process(self, normalizations=None, proj_and_gt=None, reproject_path=None):
        self.red_ds = open_with_reproject(self.red_path, proj_and_gt, reproject_path)
        red, self.geotransform, self.projection = (self.red_ds.ReadAsArray(), self.red_ds.GetGeoTransform(),
                                                   self.red_ds.GetProjection())
        self.nir_ds = open_with_reproject(self.nir_path, proj_and_gt, reproject_path)
        nir, self.geotransform, self.projection = (self.nir_ds.ReadAsArray(), self.nir_ds.GetGeoTransform(),
                                                   self.nir_ds.GetProjection())
        self.mask_ds = open_with_reproject(self.mask_path, proj_and_gt, reproject_path,
                                           (self.nir_ds.RasterXSize, self.nir_ds.RasterYSize))
        mask, self.geotransform, self.projection = (self.mask_ds.ReadAsArray(), self.mask_ds.GetGeoTransform(),
                                                    self.mask_ds.GetProjection())
        mask = mask > self.params.THRESHOLD

        red_stats = get_stats(red, mask)
        nir_stats = get_stats(nir, mask)

        norm_coefs = get_norm_coefs(red_stats, nir_stats, normalizations)

        normalized = [linear_transform(red, nir, mask, coefs) for coefs in norm_coefs]

        r, n, m = (np.array([q[0] for q in normalized]),
                   np.array([q[1] for q in normalized]),
                   np.array([q[2] for q in normalized]))

        return r, n, m


def run(y1, y2, params):
    scene_names = os.listdir(params.MASKS_PATH)
    scene_names = [s[:-4] for s in scene_names if s.endswith('.tif')]
    scene_names = sorted(scene_names)

    result_data = ResultData()
    scenes = []

    for s in scene_names:
        print(s)
        y, m, d = map(int, (s[1:5], s[6:8], s[8:10]))
        if not y1 <= y <= y2:
            print('Excluded by year!')
            continue

        scenes.append(Scene(s, params))
        r, n, m = scenes[-1].process(params.NORMALIZATIONS, (scenes[0].projection, scenes[0].geotransform),
                                     params.REPROJECT_PATH)
        g = scenes[-1].geotransform
        result_data.process_scene(r, n, m, g)

        # save_file(f'C:/Tmp/out/result_after_{s}.tif', np.random.random(result_data.num[0].shape),
        #           result_data.geo_transform, scenes[0].projection)

    result_data.calc_coefficients(min_for_data=params.MIN_FOR_ELLIPSE)

    if not os.path.exists(params.OUTPUT_PATH):
        os.makedirs(params.OUTPUT_PATH)
    if (params.PLOTS not in ['show', 'none']) and not os.path.exists(params.PLOTS):
        os.makedirs(params.PLOTS)

    if params.ENABLE_INTERACTIVE_MODE:
        interactive_mode(scenes, result_data, params)

    geo_transform = result_data.geo_transform
    projection = scenes[0].projection

    for i, norm in enumerate(params.NORMALIZATIONS):
        axywh = result_data.axywh[i]
        a, x, y, w, h = tuple(axywh.transpose([2, 0, 1]))
        num = result_data.num[i]

        save_file(f'{params.OUTPUT_PATH}/A_norm_{norm}.tif', a, geo_transform, projection)
        save_file(f'{params.OUTPUT_PATH}/X_norm_{norm}.tif', x, geo_transform, projection)
        save_file(f'{params.OUTPUT_PATH}/Y_norm_{norm}.tif', y, geo_transform, projection)
        save_file(f'{params.OUTPUT_PATH}/W_norm_{norm}.tif', w, geo_transform, projection)
        save_file(f'{params.OUTPUT_PATH}/H_norm_{norm}.tif', h, geo_transform, projection)
        save_file(f'{params.OUTPUT_PATH}/B_norm_{norm}.tif', y - a * x, geo_transform, projection)
        save_file(f'{params.OUTPUT_PATH}/C_norm_{norm}.tif', x + y, geo_transform, projection)
        save_file(f'{params.OUTPUT_PATH}/Num_norm_{norm}.tif', num, geo_transform, projection)


if __name__ == '__main__':
    load_proj()

    params_ = Namespace()
    params_.THRESHOLD = THRESHOLD
    params_.MIN_FOR_ELLIPSE = MIN_FOR_ELLIPSE
    params_.MASKS_PATH = MASKS_PATH
    params_.ORIGINAL_PATH = ORIGINAL_PATH
    params_.OUTPUT_PATH = OUTPUT_PATH
    params_.ENABLE_INTERACTIVE_MODE = ENABLE_INTERACTIVE_MODE
    params_.NORMALIZATIONS = NORMALIZATIONS
    params_.PLOTS = PLOTS
    params_.REPROJECT_PATH = REPROJECT_PATH

    parser = ArgumentParser()
    parser.add_argument('--json', type=str, default='soilline_coefficients_config.json')
    options = vars(parser.parse_args())
    json_path = options['json']

    override = json.load(open(json_path))
    params_.update(**override)

    y1, y2 = -10000, 10000

    while True:
        try:
            time_period = input('Enter time period -- y1 and y2 -- or just press <Enter> :>')
            if not len(time_period.split()):
                break
            if len(time_period.split()) == 2:
                y1, y2 = map(int, time_period.split())
                break
        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            time.sleep(.2)
            print('\n\n', e)
            continue

    run(y1, y2, params=params_)
