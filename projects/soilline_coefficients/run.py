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
    Namespace, align_images, logger


class ResultData:
    def __init__(self):
        self.channels = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']
        self.sums_keys = self.channels + [c + '_squared' for c in self.channels] + ['red_nir', 'swir1_swir2', 'num']
        self._sums = dict()
        self.res = None

        self.geo_transform = None

    def process_scene(self, channel_values, mask, geo_transform):
        """
        Processes a split scene -- firstly merges pieces, then updates statistics.
        channel_values: [{channel: value} for norm]
        """
        channel_values = {key: np.array([value[key] for value in channel_values]) for key in channel_values[0]}
        update_sums = dict()
        for channel in self.channels:
            update_sums[channel] = channel_values[channel]
            update_sums[channel + '_squared'] = channel_values[channel] ** 2
        update_sums['red_nir'] = channel_values['red'] * channel_values['nir']
        update_sums['swir1_swir2'] = channel_values['swir1'] * channel_values['swir2']
        update_sums['num'] = np.array([mask.astype(int) for _ in range(len(channel_values['red']))])

        if not self._sums:
            for key in update_sums:
                self._sums[key] = update_sums[key]
            self.geo_transform = geo_transform
        else:
            if True:  # list(geo_transform) != list(self.geo_transform):
                a = tuple(self._sums[key] for key in self.sums_keys)
                b = tuple(update_sums[key] for key in self.sums_keys)

                a, b, self.geo_transform = align_images(a, b, self.geo_transform, geo_transform)

                for i, key in enumerate(self.sums_keys):
                    self._sums[key] = a[i]
                    update_sums[key] = b[i]
            for key in self.sums_keys:
                self._sums[key] += update_sums[key]

    def calc_coefficients(self, min_for_data: float):
        if not self._sums:
            raise ValueError('Probably there were no processed scenes!')
        with np.errstate(all='ignore'):
            res = dict()
            for key in self.channels:
                res[key + '_mean'] = self._sums[key] / self._sums['num']
                res[key + '_disp'] = self._sums[key + '_squared'] / self._sums['num'] - res[key + '_mean'] ** 2

            # ellipse params for red-nir
            red_nir = np.nan_to_num(self._sums['red_nir'] / self._sums['num']) - res['red_mean'] * res['nir_mean']
            res['alpha'] = (res['red_disp'] - res['nir_disp'] +
                            ((res['red_disp'] - res['nir_disp']) ** 2 + 4 * red_nir ** 2) ** .5) / red_nir / 2
            res['width'] = (res['red_disp'] + red_nir / res['alpha']) ** .5
            res['height'] = (res['red_disp'] - red_nir * res['alpha']) ** .5
            res['angle_tg'] = 1 / res['alpha']
            res['red_plus_nir_mean'] = res['red_mean'] + res['nir_mean']
            res['bias'] = res['nir_mean'] - res['red_mean'] / res['alpha']

            # ellipse params for swir1-swir2
            swir1_swir2 = np.nan_to_num(self._sums['swir1_swir2'] / self._sums['num']) - \
                          res['swir1_mean'] * res['swir2_mean']
            res['sw12_alpha'] = (res['swir1_disp'] - res['swir2_disp'] + ((res['swir1_disp'] - res['swir2_disp']) ** 2 +
                                                                          4 * swir1_swir2 ** 2) ** .5) / swir1_swir2 / 2
            res['sw12_width'] = (res['swir1_disp'] + swir1_swir2 / res['sw12_alpha']) ** .5
            res['sw12_height'] = (res['swir1_disp'] - swir1_swir2 * res['sw12_alpha']) ** .5
            res['sw12_angle_tg'] = 1 / res['sw12_alpha']
            res['sw12_red_plus_nir_mean'] = res['swir1_mean'] + res['swir2_mean']
            res['sw12_bias'] = res['swir2_mean'] - res['swir1_mean'] / res['sw12_alpha']

            for key in res.keys():
                res[key] = np.nan_to_num(res[key]) * (self._sums['num'] >= min_for_data)
            res['num'] = self._sums['num']
            self.res = res


class ScenePiece:
    def __init__(self, scene_name, params):
        self.channels = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']
        sat_ch = SATELLITE_CHANNELS[[s for s in SATELLITE_CHANNELS.keys() if s in scene_name][0]]
        sat_ch = {v: k for k, v in sat_ch.items()}
        self.scene_name = scene_name
        self.params = params

        self.file_paths = {key: os.path.join(params.ORIGINAL_PATH, scene_name + f'_{key}_{sat_ch[key]}.tif')
                           for key in self.channels}
        self.mask_path = os.path.join(params.MASKS_PATH, scene_name + '.tif')
        self.ds = dict()
        self.mask_ds = None

        ds = gdal.Open(self.file_paths['red'])
        if ds is None:
            raise FileNotFoundError(self.file_paths['red'])
        self.geotransform = ds.GetGeoTransform()
        self.projection = ds.GetProjection()

    def process(self, normalizations=None, proj_and_gt=None, reproject_path=None):
        channel_value = dict()
        stats = dict()
        self.projection = None
        self.geotransform = None
        for i, key in enumerate(self.channels):
            self.ds[key] = open_with_reproject(self.file_paths[key], proj_and_gt, reproject_path)
            if not self.geotransform:
                self.geotransform = self.ds[key].GetGeoTransform()
            if self.file_paths[key] is None:
                raise FileNotFoundError(self.file_paths[key])
            channel_value[key] = self.ds[key].ReadAsArray()

            if self.projection and not (self.projection == self.ds[key].GetProjection()):
                raise NotImplementedError('All channels and Mask should have equal projections!')

            if self.geotransform and not (self.geotransform == self.ds[key].GetGeoTransform() and
                                          channel_value[key].shape == channel_value['red'].shape):
                logger.info('Align channels')
                new_channel_values, (channel_value[key],), self.geotransform = \
                    align_images(tuple(channel_value[k] for k in self.channels[:i]), (channel_value[key],),
                                 self.geotransform, self.ds[key].GetGeoTransform())
                for j, k in enumerate(self.channels[:i]):
                    channel_value[k] = new_channel_values[j]

        self.mask_ds = open_with_reproject(self.mask_path, proj_and_gt, reproject_path,
                                           (self.ds['red'].RasterXSize, self.ds['nir'].RasterYSize))
        mask = self.mask_ds.ReadAsArray()
        if not (self.geotransform == self.mask_ds.GetGeoTransform() and
                mask.shape == channel_value['red'].shape):
            logger.info('Align channels with mask')
            new_channel_values, (mask,), self.geotransform = \
                align_images(tuple(channel_value[k] for k in self.channels), (mask,),
                             self.geotransform, self.mask_ds.GetGeoTransform())
            for j, k in enumerate(self.channels):
                channel_value[k] = new_channel_values[j]

        mask = np.array(mask > self.params.THRESHOLD)

        for key in self.channels:
            stats[key] = get_stats(channel_value[key], mask)

        norm_coefs = get_norm_coefs(stats, normalizations)

        normalized = [linear_transform(channel_value, mask, coefs) for coefs in norm_coefs]

        return normalized, mask


class Scene:
    def __init__(self, name, pieces):
        self.channels = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']
        self.scene_name = name
        self.pieces = pieces
        self.geotransform = pieces[0].geotransform
        self.projection = pieces[0].projection

    def get_rnm(self, normalizations=None, proj_and_gt=None, reproject_path=None):
        values_and_masks = [p.process(normalizations, proj_and_gt, reproject_path) for p in self.pieces]
        # [([{value for channel} for norm], mask) for piece]

        geo_transforms = [p.geotransform for p in self.pieces]

        channel_value, mask = values_and_masks[0]
        geo_transform = geo_transforms[0]
        num = mask.astype(int)
        for (cv, m), gt in list(zip(values_and_masks, geo_transforms))[1:]:
            res = None
            upd = None
            for norm in range(len(channel_value)):
                res = tuple(channel_value[norm][key] for key in self.channels) + tuple([num])
                upd = tuple(cv[norm][key] for key in self.channels) + tuple([m])
                res, upd, geo_transform = align_images(res, upd, geo_transform, gt)

                for i, key in enumerate(self.channels):
                    channel_value[norm][key] = res[i] + upd[i]
            num = res[-1] + upd[-1]

        mask = (num >= 1).astype(int)
        for norm in range(len(channel_value)):
            for key in self.channels:
                channel_value[norm][key] /= np.maximum(num, 1)

        self.geotransform = geo_transform
        self.projection = proj_and_gt[0]

        return channel_value, mask, geo_transform


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
                channel_value, m, gt = scenes[-1].get_rnm(params.NORMALIZATIONS,
                                                          (scenes[0].projection, scenes[0].geotransform),
                                                          params.REPROJECT_PATH)
            except Exception as ex:
                logger.exception(ex)
                logger.info('Scene skipped!')
                continue

            g = scenes[-1].geotransform
            result_data.process_scene(channel_value, m, g)

    result_data.calc_coefficients(min_for_data=params.MIN_FOR_ELLIPSE)

    if not os.path.exists(params.OUTPUT_PATH):
        os.makedirs(params.OUTPUT_PATH)
    if not os.path.exists(os.path.join(params.OUTPUT_PATH, 'uint8')):
        os.makedirs(os.path.join(params.OUTPUT_PATH, 'uint8'))
    if (params.PLOTS not in ['show', 'none']) and not os.path.exists(params.PLOTS):
        os.makedirs(params.PLOTS)

    if params.ENABLE_INTERACTIVE_MODE:
        interactive_mode(scenes, result_data, params)

    geo_transform = result_data.geo_transform
    projection = scenes[0].projection

    for i, norm in enumerate(params.NORMALIZATIONS):
        for key, value in result_data.res.items():
            save_file(f'{params.OUTPUT_PATH}/{key}_norm_{norm}.tif', value[i], geo_transform, projection)

            arr = value[i]
            arr_min = np.quantile(arr[np.abs(arr) > 1e-6], .001)
            arr_max = np.quantile(arr[np.abs(arr) > 1e-6], .999)
            arr = (arr - arr_min) / (arr_max - arr_min) * 256
            arr[arr < 1e-6] = 0
            arr[arr > 255 - 1e-6] = 255
            arr = arr.astype(np.uint8)
            save_file(f'{params.OUTPUT_PATH}/uint8/{key}_norm_{norm}.tif', arr, geo_transform, projection,
                      dtype=gdal.GDT_Byte)


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
