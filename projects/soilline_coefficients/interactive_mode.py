import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from sc_utils import get_align_coefficients, logger


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
            input_x, input_y = v.split()
            input_x, input_y = int(input_x), int(input_y)

            x, y = result_data.res['num'].shape[1] - 1 - input_y, input_x

            coordinates = (result_data.geo_transform[0] + y * result_data.geo_transform[1],
                           result_data.geo_transform[3] + x * result_data.geo_transform[5])
            logger.info(f'Coordinates: {coordinates}')
            print(f'num={result_data.res["num"][0, x, y]}',
                  {key: value[0, x, y] for (key, value) in result_data.res.items()})
            if result_data.res['num'][0, x, y] == 0:
                continue

            plt.clf()
            fig = plt.figure(0)
            ax = fig.add_subplot(111, aspect='equal')

            e = None
            if result_data.res['num'][0, x, y] > 7:
                e = Ellipse(xy=(result_data.res['red_mean'][0, x, y],
                                result_data.res['nir_mean'][0, x, y]),
                            width=result_data.res['width'][0, x, y] * 5,
                            height=result_data.res['height'][0, x, y] * 5,
                            angle=np.arctan(result_data.res['angle_tg'][0, x, y]) / np.pi * 180,
                            edgecolor='red',
                            facecolor='none')
                ax.add_artist(e)

            stats = []
            valid_scenes = []
            for s in scenes:
                flag_found_piece = False
                for p in s.pieces:
                    if flag_found_piece:
                        break

                    p_shape = p.ds['red'].RasterYSize, p.ds['red'].RasterXSize
                    x_, y_, dx, dy, udlr1, udlr2, gt = \
                        get_align_coefficients(result_data.geo_transform, p.geotransform,
                                               result_data.res['num'].shape[-2:], p_shape)

                    x_shift, y_shift = x - udlr2[0], y - udlr2[2]
                    assert sum(udlr1) == 0

                    if x_shift >= p.mask_ds.RasterYSize or y_shift >= p.mask_ds.RasterXSize:
                        continue

                    if p.mask_ds.ReadAsArray(y_shift, x_shift, 1, 1)[0, 0] > params.THRESHOLD:
                        stats.append([p.ds['red'].ReadAsArray(y_shift, x_shift, 1, 1)[0, 0],
                                      p.ds['nir'].ReadAsArray(y_shift, x_shift, 1, 1)[0, 0]])
                        valid_scenes.append(s.scene_name)
                        flag_found_piece = True
            stats = np.array(stats)

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

            logger.info(f'Found {len(valid_scenes)}, red_range = {red_range}, nir_range = {nir_range}')
            if params.PLOTS == 'show':
                plt.show()
            elif params.PLOTS != 'none':
                plot_path = params.PLOTS + f'/plot_{input_x}_{input_y}.png'
                plt.savefig(plot_path)
                logger.info(f'Saved plot to {plot_path}')

                out_stats = {'num': int(result_data.res['num'][0, x, y])}
                for key, value in result_data.res.items():
                    out_stats[key] = float(value[0, x, y])
                out_stats['points'] = stats.tolist()
                out_stats['scenes'] = valid_scenes

                stats_path = params.PLOTS + f'/stats_{input_x}_{input_y}.json'
                with open(stats_path, 'w') as f:
                    json.dump(out_stats, f, indent=4)
                logger.info(f'Saved stats to {stats_path}')

        except Exception as e:
            logger.exception(e)
            continue
