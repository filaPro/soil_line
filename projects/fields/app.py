import os
import json
import numpy as np
import pandas as pd
from osgeo import ogr, gdal
from argparse import ArgumentParser

from lib import reshape_points, make_cropped_mask, erode, dilate, save


def make_mask(points, resolution):
    points = reshape_points(points)
    concatenated = np.concatenate([np.concatenate(p) for p in points])
    x_min = concatenated[:, 0].min() - resolution
    y_max = concatenated[:, 1].max() + resolution
    mask_width = concatenated[:, 0].max() - x_min + resolution
    mask_height = y_max - concatenated[:, 1].min() + resolution
    width = int(mask_width // resolution + 1)
    height = int(mask_height // resolution + 1)
    mask = make_cropped_mask(points, resolution, x_min, y_max, height, width)
    return mask, x_min, y_max, mask_width, mask_height


def crop(tif_file, x_mask_min, y_mask_max, mask_width, mask_height, resolution):
    x_min, _, _, y_max, _, _ = tif_file.GetGeoTransform()
    image = tif_file.GetRasterBand(1).ReadAsArray()
    x_crop_min = int((x_mask_min - x_min) // resolution)
    x_crop_max = int((x_mask_min - x_min) // resolution) + int(mask_width // resolution + 1)
    y_crop_min = int((y_max - y_mask_max) // resolution)
    y_crop_max = int((y_max - y_mask_max) // resolution) + int(mask_height // resolution + 1)
    return image[y_crop_min: y_crop_max, x_crop_min: x_crop_max]


def load_and_crop_images(
    excel_file, name, tmp_path, tif_path, spatial_reference, x_mask_min, y_mask_max, mask_width, mask_height
):
    images = []
    for _, row in excel_file[excel_file['name'] == name].iterrows():
        tif_file_name = row['NDVI_map'][:-4] + '.tif'
        print(os.path.join(tif_path, tif_file_name))
        tif_file = gdal.Open(os.path.join(tif_path, tif_file_name))
        tif_file = gdal.Warp(
            destNameOrDestDS=tmp_path,
            srcDSOrSrcDSTab=tif_file,
            dstSRS=spatial_reference,
            dstNodata=0,
            xRes=options['resolution'],
            yRes=options['resolution'],
            resampleAlg='cubic'
        )
        image = crop(
            tif_file=tif_file,
            x_mask_min=x_mask_min,
            y_mask_max=y_mask_max,
            mask_width=mask_width,
            mask_height=mask_height,
            resolution=options['resolution']
        )
        images.append(image)
    return images


def compute_deviation(images, mask):
    deviations = np.empty((len(images), mask.shape[0], mask.shape[1]))
    deviations.fill(np.nan)
    for i in range(len(images)):
        deviation = images[i]
        deviation[np.where(np.logical_not(mask))] = np.nan
        deviation -= np.nanmean(deviation)
        deviations[i] = deviation
    return deviations


def apply_quantiles(deviation, min_quantile, max_quantile):
    if min_quantile > .0:
        min_threshold = np.nanquantile(deviation, min_quantile)
        deviation[deviation < min_threshold] = np.nan
    if max_quantile < 1.:
        max_threshold = np.nanquantile(deviation, max_quantile)
        deviation[deviation > max_threshold] = np.nan
    return deviation


def aggregate(deviations, method):
    if method == 'min':
        return np.nanmin(deviations, axis=0)
    elif method == 'mean':
        return np.nanmean(deviations, axis=0)
    elif method == 'max':
        return np.nanmax(deviations, axis=0)
    elif method == 'max_minus_min':
        return np.nanmax(deviations, axis=0) - np.nanmin(deviations, axis=0)
    else:
        raise ValueError(f'Invalid aggregation_method: {method}')



def dilate_images(images, mask, buffer_size, fill_method, tmp_path):
    for i in range(len(images)):
        images[i][np.logical_not(mask)] = np.nan
        images[i] = dilate(images[i], buffer_size, fill_method, tmp_path)


def run_field(
    field, spatial_reference, tmp_path, buffer_size, resolution, min_quantile, max_quantile, fill_method, tif_path,
    excel_file, out_path, aggregation_method
):
    points = field['geometry']['coordinates']
    name = field['properties']['name']
    print(name)
    full_mask, x_mask_min, y_mask_max, mask_width, mask_height = make_mask(points, resolution)
    mask = erode(full_mask, buffer_size)
    images = load_and_crop_images(
        excel_file=excel_file,
        name=name,
        tmp_path=tmp_path,
        tif_path=tif_path,
        spatial_reference=spatial_reference,
        x_mask_min=x_mask_min,
        y_mask_max=y_mask_max,
        mask_width=mask_width,
        mask_height=mask_height
    )
    dilate_images(images, mask, buffer_size, fill_method, tmp_path)
    deviations = compute_deviation(images, full_mask)
    deviation = aggregate(deviations, aggregation_method)
    deviation = apply_quantiles(deviation, min_quantile, max_quantile)
    deviation[np.where(np.logical_not(full_mask))] = -1
    deviation[np.where(np.isnan(deviation))] = -1
    save(
        deviation=deviation,
        out_path=out_path,
        name=name,
        spatial_reference=spatial_reference,
        x_mask_min=x_mask_min,
        y_mask_max=y_mask_max,
        resolution=resolution
    )


def run(
    tmp_path, buffer_size, resolution, min_quantile, max_quantile, fill_method, tif_path, shape_path,
    excel_path, out_path, aggregation_method
):
    shape_file = ogr.Open(shape_path)
    excel_file = pd.read_excel(excel_path)
    os.makedirs(out_path, exist_ok=True)
    layer = shape_file.GetLayer(0)
    for feature in layer:
        run_field(
            field=json.loads(feature.ExportToJson()),
            spatial_reference=layer.GetSpatialRef(),
            tmp_path=tmp_path,
            buffer_size=buffer_size,
            resolution=resolution,
            min_quantile=min_quantile,
            max_quantile=max_quantile,
            fill_method=fill_method,
            tif_path=tif_path,
            excel_file=excel_file,
            out_path=out_path,
            aggregation_method=aggregation_method
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_path', type=str, default='/volume')
    parser.add_argument('--tmp_path', type=str, default='/tmp/tmp.tif')
    parser.add_argument('--buffer_size', type=int, default=0)
    parser.add_argument('--resolution', type=float, default=10.)
    parser.add_argument('--min_quantile', type=float, default=.0)
    parser.add_argument('--max_quantile', type=float, default=1.)
    parser.add_argument('--fill_method', type=str, default='ns')
    parser.add_argument('--aggregation_method', type=str, default='mean')

    options = vars(parser.parse_args())
    in_path = options['in_path']
    run(
        tmp_path=options['tmp_path'],
        buffer_size=options['buffer_size'],
        resolution=options['resolution'],
        min_quantile=options['min_quantile'],
        max_quantile=options['max_quantile'],
        fill_method=options['fill_method'],
        tif_path=os.path.join(in_path, 'NDVI_tif'),
        shape_path=os.path.join(in_path, 'fields.shp'),
        excel_path=os.path.join(in_path, 'NDVI_list.xls'),
        out_path=os.path.join(in_path, 'out', 'deviations'),
        aggregation_method=options['aggregation_method']
    )
