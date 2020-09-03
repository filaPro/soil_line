import sys
import time
import traceback
import numpy as np
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sc_utils import get_transform_coefficients


def interactive_mode(scenes, result_data, params):
    while True:
        v = input('\nEnter coordinates or type "Q" or "S": >>>')
        if v == 'Q':
            print('\n\nQuit!')
            return
        if v == 'S':
            print('\n\nFinishing interactive mode, now exporting data.')
            break
        try:
            coord_x, coord_y = v.split()
            coord_x, coord_y = int(coord_x), int(coord_y)

            x, y = result_data.num.shape[1] - 1 - coord_y, coord_x

            print(f'num={result_data.num[0, x, y]}, AXYWH = {result_data.axywh[0, x, y]}')
            if result_data.num[0, x, y] == 0:
                continue

            plt.clf()
            fig = plt.figure(0)
            ax = fig.add_subplot(111, aspect='equal')

            e = None
            if result_data.num[0, x, y] > 7:
                e = Ellipse(xy=(result_data.axywh[0, x, y, 1],
                                result_data.axywh[0, x, y, 2]),
                            width=result_data.axywh[0, x, y, 3] * 5,
                            height=result_data.axywh[0, x, y, 4] * 5,
                            angle=np.arctan(result_data.axywh[0, x, y, 0]) / np.pi * 180,
                            edgecolor='red',
                            facecolor='none')
                ax.add_artist(e)

            stats = []
            valid_scenes = []
            for s in scenes:
                x_, y_, dx, dy = get_transform_coefficients(result_data.geo_transform, s.geotransform)

                l1, r1, u1, d1 = (max(-x_, 0), max(x_ + s.mask_ds.RasterYSize - result_data.sum_red.shape[1], 0),
                                  max(-y_, 0), max(y_ + s.mask_ds.RasterXSize - result_data.sum_red.shape[2], 0))
                l2, r2, u2, d2 = (max(x_, 0), max(-x_ - s.mask_ds.RasterYSize + result_data.sum_red.shape[1], 0),
                                  max(y_, 0), max(-y_ - s.mask_ds.RasterXSize + result_data.sum_red.shape[2], 0))

                assert l1 + r1 + u1 + d1 == 0

                x_shift, y_shift = x - l2, y - u2

                # print(l1, r1, u1, d1, l2, r2, u2, d2, x_shift, y_shift, s.mask_ds.RasterYSize, s.mask_ds.RasterXSize)
                # print(s.mask_ds.ReadAsArray(y_shift, x_shift, 1, 1)[0, 0])
                # print(s.mask_ds.ReadAsArray()[y_shift, x_shift])

                if x_shift >= s.mask_ds.RasterYSize or y_shift >= s.mask_ds.RasterXSize:
                    continue

                if s.mask_ds.ReadAsArray(y_shift, x_shift, 1, 1)[0, 0] > params.THRESHOLD:
                    stats.append([s.red_ds.ReadAsArray(y_shift, x_shift, 1, 1)[0, 0],
                                  s.nir_ds.ReadAsArray(y_shift, x_shift, 1, 1)[0, 0]])
                    valid_scenes.append(s.scene_name)
            stats = np.array(stats)
            print(len(valid_scenes), result_data.num[0, x, y])

            plt.scatter(*tuple(stats.transpose()))

            red_range = stats[:, 0].min(), stats[:, 0].max()
            red_range = (red_range[0] + (red_range[0] - red_range[1]) * .05,
                         red_range[1] + (red_range[1] - red_range[0]) * .05)
            nir_range = stats[:, 1].min(), stats[:, 1].max()
            nir_range = (nir_range[0] + (nir_range[0] - nir_range[1]) * .05,
                         nir_range[1] + (nir_range[1] - nir_range[0]) * .05)
            if e:
                ext = (e.center[0] - e.width / 2, e.center[1] - e.width / 2,
                       e.center[0] + e.width / 2, e.center[1] + e.width / 2)  # e.axes.dataLim.extents
                # print(ext)
                red_range = (min(red_range[0], ext[0]), max(red_range[1], ext[2]))
                nir_range = (min(nir_range[0], ext[1]), max(nir_range[1], ext[3]))

            ax.set_xlim(*red_range)
            ax.set_ylim(*nir_range)

            ax.set_xlabel('red')
            ax.set_ylabel('nir')

            if params.PLOTS == 'show':
                plt.show()
            elif params.PLOTS != 'none':
                plt.savefig(params.PLOTS + f'/plot_{coord_x}_{coord_y}.png')

                out_stats = {'num': int(result_data.num[0, x, y])}
                for i, k in enumerate('AXYWH'):
                    out_stats[k] = float(result_data.axywh[0, x, y, i])
                out_stats['points'] = stats.tolist()
                out_stats['scenes'] = valid_scenes

                with open(params.PLOTS + f'/stats_{coord_x}_{coord_y}.json', 'w') as f:
                    json.dump(out_stats, f, indent=4)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            time.sleep(.2)
            print('\n\n', e)
            continue
