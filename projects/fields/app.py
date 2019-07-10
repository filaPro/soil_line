import os
import cv2
import json
import numpy as np
import pandas as pd
from osgeo import ogr, gdal
from argparse import ArgumentParser


def make_mask(points, resolution):
    depth = lambda l: isinstance(l, list) and (max(map(depth, l)) + 1) if l else 1
    if depth(points) == 3:
        points = [points]
    concatenated = np.concatenate([np.concatenate(p) for p in points])
    x_min = concatenated[:, 0].min()
    y_max = concatenated[:, 1].max()
    mask_width = concatenated[:, 0].max() - x_min
    mask_height = y_max - concatenated[:, 1].min()
    width = int(mask_width // resolution + 1)
    height = int(mask_height // resolution + 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    for subfield in points:
        for i, contour in enumerate(subfield):
            contour = np.array(contour)
            contour[:, 0] = (contour[:, 0] - x_min) // resolution
            contour[:, 1] = (y_max - contour[:, 1]) // resolution
            color = 1 if i == 0 else 0
            cv2.fillPoly(mask, np.reshape(contour, (1, len(contour), 2)).astype(np.int32), color)
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


def crop(tif_file, x_mask_min, y_mask_max, mask_width, mask_height, resolution):
    x_min, _, _, y_max, _, _ = tif_file.GetGeoTransform()
    image = tif_file.GetRasterBand(1).ReadAsArray()
    x_crop_min = int((x_mask_min - x_min) // resolution)
    x_crop_max = int((x_mask_min - x_min) // resolution) + int(mask_width // resolution + 1)
    y_crop_min = int((y_max - y_mask_max) // resolution)
    y_crop_max = int((y_max - y_mask_max) // resolution) + int(mask_height // resolution + 1)
    return image[y_crop_min: y_crop_max, x_crop_min: x_crop_max]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_path', type=str, default='/volume')
    parser.add_argument('--tmp_path', type=str, default='/tmp/tmp.tif')
    parser.add_argument('--buffer_size', type=int, default=0)
    parser.add_argument('--resolution', type=float, default=10.0)
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
        full_mask, x_mask_min, y_mask_max, mask_width, mask_height = make_mask(
            points, resolution=options['resolution']
        )
        mask = erode(full_mask, n_iterations=options['buffer_size'])
        images = []
        for _, row in excel_file[excel_file['name'] == name].iterrows():
            tif_file_name = row['NDVI_map'][:-4] + '.tif'
            print(os.path.join(tif_path, tif_file_name))
            tif_file = gdal.Open(os.path.join(tif_path, tif_file_name))
            tif_file = gdal.Warp(
                '/tmp/tmp.tif', tif_file, dstSRS=layer.GetSpatialRef(), dstNodata=0,
                xRes=options['resolution'], yRes=options['resolution'], resampleAlg='cubic'
            )
            image = crop(
                tif_file=tif_file, x_mask_min=x_mask_min, y_mask_max=y_mask_max,
                mask_width=mask_width, mask_height=mask_height, resolution=options['resolution']
            )
            images.append(image)
        deviations = np.empty((len(images), mask.shape[0], mask.shape[1]))
        deviations.fill(np.nan)
        for i in range(len(images)):
            deviation = images[i]
            deviation[np.where(np.logical_not(mask))] = np.nan
            deviation -= np.nanmean(deviation)
            deviations[i] = deviation
        deviation = np.nanmean(deviations, axis=0)
        if options['buffer_size'] > 0:
            deviation = cv2.inpaint(
                deviation.astype(np.float32), np.isnan(deviation).astype(np.uint8), 1, cv2.INPAINT_NS
            )
            deviation[np.where(np.logical_not(full_mask))] = -1
        deviation[np.where(np.isnan(deviation))] = -1

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            os.path.join(out_path, f'{name}.tif'), mask.shape[1], mask.shape[0], 1, gdal.GDT_Float32
        )
        dataset.SetGeoTransform((x_mask_min, options['resolution'], 0, y_mask_max, 0, -options['resolution']))
        dataset.SetProjection(tif_file.GetSpatialRef().ExportToWkt())
        dataset.GetRasterBand(1).WriteArray(deviation)
        dataset.FlushCache()
