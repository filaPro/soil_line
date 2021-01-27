import json
import os
from argparse import ArgumentParser

import gdal
import numpy as np

from config import THRESHOLD, MASKS_PATH, ORIGINAL_PATH, OUTPUT_PATH, ENABLE_INTERACTIVE_MODE, NORMALIZATIONS, PLOTS, \
    MIN_FOR_ELLIPSE, REPROJECT_PATH
from interactive_mode import interactive_mode
from reproject import open_with_reproject
from sc_utils import SATELLITE_CHANNELS, get_stats, get_norm_coefs, linear_transform, save_file, \
    Namespace, load_proj, align_images, logger


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
        """
        Processes a split scene -- firstly merges pieces, then updates statistics.
        """
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
                a = (self.sum_red, self.sum_nir, self.sum_red2, self.sum_nir2, self.sum_red_nir, self.num)
                b = (red, nir, red2, nir2, red_nir, mask)

                a, b, self.geo_transform = align_images(a, b, self.geo_transform, geo_transform)

                (self.sum_red, self.sum_nir, self.sum_red2, self.sum_nir2, self.sum_red_nir, self.num) = a
                (red, nir, red2, nir2, red_nir, mask) = b
            self.sum_red += red
            self.sum_nir += nir
            self.sum_red2 += red2
            self.sum_nir2 += nir2
            self.sum_red_nir += red_nir
            self.num += mask.astype(int)

    def calc_coefficients(self, min_for_data: float):
        if self.num is None:
            raise ValueError('Probably there were no processed scenes!')
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


class ScenePiece:
    def __init__(self, scene_name, params):
        sat_ch = SATELLITE_CHANNELS[[s for s in SATELLITE_CHANNELS.keys() if s in scene_name][0]]
        sat_ch = {v: k for k, v in sat_ch.items()}
        self.scene_name = scene_name
        self.params = params
        self.red_path = os.path.join(params.ORIGINAL_PATH, scene_name + f'_red_{sat_ch["red"]}.tif')
        self.nir_path = os.path.join(params.ORIGINAL_PATH, scene_name + f'_nir_{sat_ch["nir"]}.tif')
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
        nir, nir_geotransform, nir_projection = (self.nir_ds.ReadAsArray(), self.nir_ds.GetGeoTransform(),
                                                 self.nir_ds.GetProjection())
        self.mask_ds = open_with_reproject(self.mask_path, proj_and_gt, reproject_path,
                                           (self.nir_ds.RasterXSize, self.nir_ds.RasterYSize))
        mask, mask_geotransform, mask_projection = (self.mask_ds.ReadAsArray(), self.mask_ds.GetGeoTransform(),
                                                    self.mask_ds.GetProjection())
        mask = np.array(mask > self.params.THRESHOLD)

        if not (self.projection == nir_projection == mask_projection):
            raise NotImplementedError('Red, Nir, Mask should have equal projections!')
        if not (self.geotransform == nir_geotransform and red.shape == nir.shape):
            logger.warning('Align red and nir')
            (red,), (nir,), self.geotransform = align_images((red,), (nir,), self.geotransform, nir_geotransform)
        if not (self.geotransform == mask_geotransform and red.shape == mask.shape):
            logger.info('Align red-nir and mask')
            (red, nir), (mask,), self.geotransform = \
                align_images((red, nir), (mask,), self.geotransform, mask_geotransform)

        red_stats = get_stats(red, mask)
        nir_stats = get_stats(nir, mask)

        norm_coefs = get_norm_coefs(red_stats, nir_stats, normalizations)

        normalized = [linear_transform(red, nir, mask, coefs) for coefs in norm_coefs]

        r, n, m = (np.array([q[0] for q in normalized]),
                   np.array([q[1] for q in normalized]),
                   np.array([q[2] for q in normalized]))

        return r, n, m


class Scene:
    def __init__(self, name, pieces):
        self.scene_name = name
        self.pieces = pieces
        self.geotransform = pieces[0].geotransform
        self.projection = pieces[0].projection

    def get_rnm(self, normalizations=None, proj_and_gt=None, reproject_path=None):
        rnm = [p.process(normalizations, proj_and_gt, reproject_path) for p in self.pieces]
        geo_transforms = [p.geotransform for p in self.pieces]

        (red, nir, mask), geo_transform = rnm[0], geo_transforms[0]
        num = mask.astype(int)
        for (r, n, m), gt in list(zip(rnm, geo_transforms))[1:]:
            (red, nir, num), (r, n, m), geo_transform = \
                align_images((red, nir, num), (r, n, m), geo_transform, gt)

            red += r
            nir += n
            num += m.astype(int)

        mask = (num >= 1).astype(int)
        red = red / np.maximum(num, 1)
        nir = nir / np.maximum(num, 1)

        self.geotransform = geo_transform
        self.projection = proj_and_gt[0]

        return red, nir, mask, geo_transform


def run(scenes_filter, params):
    scene_names = os.listdir(params.MASKS_PATH)
    scene_names = [s[:-4] for s in scene_names if s.endswith('.tif')]
    scene_names = sorted(scene_names)

    result_data = ResultData()
    scenes = []

    i = 0
    while i < len(scene_names):
        s = scene_names[i][:-7]
        logger.info(f'Processing scene {s}')
        y, m, d = map(int, (s[1:5], s[6:8], s[8:10]))
        sat = s[11:15]

        if 'year_from' in scenes_filter and not scenes_filter['year_from'] <= y:
            logger.info('Excluded by year!')
            i += 1
            continue
        if 'year_to' in scenes_filter and not scenes_filter['year_to'] >= y:
            logger.info('Excluded by year!')
            i += 1
            continue
        if 'satellite_type' in scenes_filter and sat not in scenes_filter['satellite_type']:
            logger.info('Excluded by satellite type!')
            i += 1
            continue

        pieces = []
        while i < len(scene_names) and scene_names[i].startswith(s):
            pieces.append(ScenePiece(scene_names[i], params))
            i += 1

        if len(pieces) != 1:
            logger.info(f'Found {len(pieces)} pieces.')

        if len(pieces):
            scenes.append(Scene(s, pieces))
            try:
                r, n, m, gt = scenes[-1].get_rnm(params.NORMALIZATIONS, (scenes[0].projection, scenes[0].geotransform),
                                                 params.REPROJECT_PATH)
            except Exception as ex:
                logger.exception(ex)
                logger.info('Scene skipped!')
                continue

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

    scene_filter = dict()

    while True:
        try:
            tp_str = input('Enter time period -- y1 and y2 -- or just press <Enter> :>')
            if not len(tp_str.split()):
                break
            if len(tp_str.split()) == 2:
                y1, y2 = map(int, tp_str.split())
                scene_filter['year_from'] = y1
                scene_filter['year_to'] = y2
                break
            logger.error('Invalid format!')
        except Exception as e:
            logger.exception(e)
            continue
    while True:
        try:
            tp_str = input('Enter satellite type (landsat or sentinel) or just press <Enter> :>')
            if not len(tp_str.split()):
                break
            if len(tp_str.split()) == 1:
                satellite_type = tp_str.split()[0]
                if satellite_type == 'sentinel':
                    scene_filter['satellite_type'] = ['S2AB']
                    break
                elif satellite_type == 'landsat':
                    scene_filter['satellite_type'] = ['LT04', 'LT05', 'LE07', 'LC08']
                    break
            if set(tp_str.split()).issubset(SATELLITE_CHANNELS.keys()):
                scene_filter['satellite_type'] = tp_str.split()
                break
            logger.error('Unknown satellite type!')
        except Exception as e:
            logger.exception(e)
            continue

    run(scene_filter, params=params_)
