# noinspection PyUnresolvedReferences
import sys
import gdal
import time
import traceback
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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

            x, y = coord_x, coord_y

            print(f'num={result_data.num[0, x, y]}, AXYWH = {result_data.axywh[0, x, y]}')

            plt.clf()
            fig = plt.figure(0)
            ax = fig.add_subplot(111, aspect='equal')

            # for x_ in range(1, 1000, 50):
            #     for y_ in range(1, 1000, 50):
            #         if result_data.num[0, x_, y_] > 7:
            #             e = Ellipse(xy=(result_data.axywh[0, x_, y_, 1],
            #                             result_data.axywh[0, x_, y_, 2]),
            #                         width=result_data.axywh[0, x_, y_, 3],
            #                         height=result_data.axywh[0, x_, y_, 4],
            #                         angle=90 - np.arctan(result_data.axywh[0, x_, y_, 0]) / np.pi * 180,
            #                         edgecolor='red',
            #                         facecolor='none')
            #             ax.add_artist(e)

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
            for s in scenes:
                if gdal.Open(s.mask_path).ReadAsArray(y, x, 1, 1)[0, 0] > params.THRESHOLD:
                    stats.append([gdal.Open(s.red_path).ReadAsArray(y, x, 1, 1)[0, 0],
                                  gdal.Open(s.nir_path).ReadAsArray(y, x, 1, 1)[0, 0]])
            stats = np.array(stats)

            plt.scatter(*tuple(stats.transpose()))

            red_range = stats[:, 0].min(), stats[:, 0].max()
            red_range = (red_range[0] + (red_range[0] - red_range[1]) * .3,
                         red_range[1] + (red_range[1] - red_range[0]) * .3)
            nir_range = stats[:, 1].min(), stats[:, 1].max()
            nir_range = (nir_range[0] + (nir_range[0] - nir_range[1]) * .3,
                         nir_range[1] + (nir_range[1] - nir_range[0]) * .3)

            ax.set_xlim(*red_range)
            ax.set_ylim(*nir_range)

            if params.PLOTS == 'show':
                plt.show()
            elif params.PLOTS is not 'none':
                plt.savefig(params.PLOTS)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            time.sleep(.2)
            print('\n\n', e)
            continue
