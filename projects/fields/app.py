import os
import cv2
import json
import numpy as np
import pandas as pd
from osgeo import ogr, gdal
from argparse import ArgumentParser


def make_full_mask(points):
    concatenated = np.concatenate(points)
    x_min = concatenated[:, 0].min()
    y_max = concatenated[:, 1].max()
    mask_width = concatenated[:, 0].max() - x_min
    mask_height = y_max - concatenated[:, 1].min()
    width = int(mask_width // 10.0 + 1)
    height = int(mask_height // 10.0 + 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(points)):
        points[i] = np.array(points[i])
        points[i][:, 0] = (points[i][:, 0] - x_min) // 10.0
        points[i][:, 1] = (points[i][:, 1] - y_max) // -10.0
        color = 1 if i == 0 else 0
        cv2.fillPoly(mask, np.reshape(points[i], (1, len(points[i]), 2)).astype(np.int32), color)
    return mask, x_min, y_max, mask_width, mask_height


def make_kernel():
    return np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)


def dilate(image, n_iterations):
    for _ in range(n_iterations):
        previous = np.copy(image)
        image = cv2.dilate(image, kernel=make_kernel())
        image[previous > 0] = previous[previous > 0]
    return image


def erode(image, n_iterations):
    return cv2.erode(image, kernel=make_kernel(), iterations=n_iterations)


def _crop(image, x_mask_min, y_mask_max, mask_width, mask_height, x_min, y_max, x_step, y_step):
    x_crop_min = int((x_mask_min - x_min) // x_step)
    x_crop_max = int((x_mask_min - x_min) // x_step) + int(mask_width // x_step + 1)
    y_crop_min = int((y_mask_max - y_max) // y_step)
    y_crop_max = int((y_mask_max - y_max) // y_step) + int(mask_height // abs(y_step) + 1)
    return image[y_crop_min: y_crop_max, x_crop_min: x_crop_max]


def crop(tif_file, x_mask_min, y_mask_max, mask_width, mask_height, sensor):
    x_min, _, _, y_max, _, _ = tif_file.GetGeoTransform()
    full_image = tif_file.GetRasterBand(1).ReadAsArray()
    if sensor[0] != 'S':
        image = _crop(full_image, x_mask_min, y_mask_max, mask_width + 30, mask_height + 30, x_min, y_max, 30, -30)
        image = cv2.resize(image, dsize=(image.shape[1] * 3, image.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
        x_min = x_min + (x_mask_min - x_min) // 30.0 * 30
        y_max = y_max + (y_mask_max - y_max) // -30.0 * -30
    else:
        image = full_image
    return _crop(image, x_mask_min, y_mask_max, mask_width, mask_height, x_min, y_max, 10, -10)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--buffer_size', type=int, default=0)
    options = vars(parser.parse_args())

    in_path = options['in_path']
    config_path = os.path.join(in_path, 'config.json')
    if os.path.exists(config_path):
        config = json.loads(config_path)
        options.update(config)
    tif_path = os.path.join(in_path, 'NDVI_tif')
    shape_path = os.path.join(in_path, 'fields.shp')
    shape_file = ogr.Open(shape_path)
    excel_path = os.path.join(in_path, 'NDVI_list.xls')
    excel_file = pd.read_excel(excel_path)
    out_path = os.path.join(in_path, 'out')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    layer = shape_file.GetLayer(0)
    for feature in layer:
        field = json.loads(feature.ExportToJson())
        points = field['geometry']['coordinates']
        name = field['properties']['name']
        print(name)
        full_mask, x_mask_min, y_mask_max, mask_width, mask_height = make_full_mask(points)
        landsat_mask = erode(full_mask, n_iterations=options['buffer_size'] * 3)
        sentinel_mask = erode(full_mask, n_iterations=options['buffer_size'])
        images = []
        sensors = []
        for _, row in excel_file[excel_file['name'] == name].iterrows():
            sensor = row['sensor']
            tif_file_name = row['NDVI_map'][:-4] + '.tif'
            tif_file = gdal.Open(os.path.join(tif_path, tif_file_name))
            image = crop(
                tif_file=tif_file, x_mask_min=x_mask_min, y_mask_max=y_mask_max,
                mask_width=mask_width, mask_height=mask_height, sensor=sensor
            )
            images.append(image)
            sensors.append(sensor)
        deviations = np.empty((len(images), full_mask.shape[0], full_mask.shape[1]))
        deviations.fill(np.nan)
        for i in range(len(images)):
            mask = sentinel_mask if sensors[i][0] == 'S' else landsat_mask
            deviation = images[i]
            deviation[np.where(np.logical_not(mask))] = np.nan
            deviation -= np.nanmean(deviation)
            deviations[i] = deviation
        deviation = np.nanmean(deviations, axis=0)
        deviation += 1
        deviation[np.isnan(deviation)] = 0
        deviation = dilate(deviation, n_iterations=options['buffer_size'] * 3)
        deviation -= 1
        deviation[np.where(np.logical_not(full_mask))] = -1

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            os.path.join(out_path, f'{name}.tif'), full_mask.shape[1], full_mask.shape[0], 1, gdal.GDT_Float32
        )
        dataset.SetGeoTransform((x_mask_min, 10, 0, y_mask_max, 0, -10))
        dataset.SetProjection(tif_file.GetSpatialRef().ExportToWkt())
        dataset.GetRasterBand(1).WriteArray(deviation)
        dataset.FlushCache()
