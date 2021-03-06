import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from osgeo import ogr, gdal

from lib import reshape_points, make_cropped_mask, erode, dilate, save, load_proj


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


def get_year(tif_file_name):
    return int(tif_file_name.split('_')[0][1:])


def load_and_crop_images(
    excel_file, name, tif_path, spatial_reference, x_mask_min, y_mask_max, mask_width, mask_height, resolution
):
    images, years = [], []
    for _, row in excel_file[excel_file['name'] == name].iterrows():
        tif_file_name = row['NDVI_map'][:-4] + '.tif'
        print(os.path.join(tif_path, tif_file_name))
        tif_file = gdal.Open(os.path.join(tif_path, tif_file_name))
        tif_file = gdal.Warp(
            destNameOrDestDS='',
            srcDSOrSrcDSTab=tif_file,
            format='VRT',
            dstSRS=spatial_reference,
            dstNodata=0,
            xRes=resolution,
            yRes=resolution,
            resampleAlg='cubic'
        )
        image = crop(
            tif_file=tif_file,
            x_mask_min=x_mask_min,
            y_mask_max=y_mask_max,
            mask_width=mask_width,
            mask_height=mask_height,
            resolution=resolution
        )
        images.append(image)
        years.append(get_year(tif_file_name))
    return images, years


def compute_and_preprocess_deviation(
    images, mask, full_mask, buffer_size, deviation_method, dilation_method, fill_method
):
    if dilation_method == 1:
        for i in range(len(images)):
            images[i][np.logical_not(mask)] = np.nan
            images[i] = dilate(images[i], buffer_size, fill_method)
        deviations = compute_deviation(images, full_mask, deviation_method)
    else:
        deviations = compute_deviation(images, mask, deviation_method)
    if dilation_method == 2:
        for i in range(len(images)):
            deviations[i][np.logical_not(mask)] = np.nan
            deviations[i] = dilate(deviations[i], buffer_size, fill_method)
    return deviations


def postprocess_deviation(deviation, buffer_size, dilation_method, fill_method):
    if dilation_method == 3:
        deviation = dilate(deviation, buffer_size, fill_method)
    return deviation


def compute_deviation(images, mask, method):
    deviations = np.empty((len(images), mask.shape[0], mask.shape[1]))
    deviations.fill(np.nan)
    for i in range(len(images)):
        deviation = images[i]
        deviation[np.where(np.logical_not(mask))] = np.nan
        if method == 'subtract_mean':
            deviation -= np.nanmean(deviation)
        elif method == 'cdf':
            quantile = np.argsort(np.argsort(deviation.flatten())).reshape(deviation.shape).astype(float)
            quantile[np.where(np.isnan(deviation))] = np.nan
            quantile = quantile / np.nanmax(quantile)
            deviation = quantile
        elif method == 'raw':
            pass
        else:
            raise ValueError(f'Invalid deviation_method: {method}')
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


def year_aggregate(deviation, year, year_method):
    if year_method == 'none':
        return deviation
    years = np.unique(year)
    deviations = np.empty((len(years), deviation.shape[1], deviation.shape[2]))
    for i, unique_year in enumerate(years):
        index = year == unique_year
        if np.sum(index) > 1:
            print(f'aggregate {np.sum(index)} deviations for year {unique_year}')
            deviations[i] = aggregate(deviation[index], year_method)
        else:
            deviations[i] = deviation[index][0]
    return deviations


def aggregate(deviations, method):
    if method == 'min':
        return np.nanmin(deviations, axis=0)
    elif method == 'mean':
        return np.nanmean(deviations, axis=0)
    elif method == 'max':
        return np.nanmax(deviations, axis=0)
    elif method == 'max_minus_min':
        return np.nanmax(deviations, axis=0) - np.nanmin(deviations, axis=0)
    elif method == 'median':
        return np.nanmedian(deviations, axis=0)
    elif method.startswith('quantile_'):
        q = float(method[9:])
        return np.nanquantile(deviations, q, axis=0)
    else:
        raise ValueError(f'Invalid aggregation_method: {method}')


def run_field(
    field, spatial_reference, buffer_size, resolution, min_quantile, max_quantile, fill_method, tif_path,
    excel_file, out_path, aggregation_method, year_aggregation_method, dilation_method, deviation_method
):
    points = field['geometry']['coordinates']
    name = field['properties']['name']
    print(name)
    full_mask, x_mask_min, y_mask_max, mask_width, mask_height = make_mask(points, resolution)
    mask = erode(full_mask, buffer_size)
    images, years = load_and_crop_images(
        excel_file=excel_file,
        name=name,
        tif_path=tif_path,
        spatial_reference=spatial_reference,
        x_mask_min=x_mask_min,
        y_mask_max=y_mask_max,
        mask_width=mask_width,
        mask_height=mask_height,
        resolution=resolution
    )
    if not len(images):
        print('skipping the field due to lack of images')
        return
    deviation = compute_and_preprocess_deviation(
        images=images,
        mask=mask,
        full_mask=full_mask,
        buffer_size=buffer_size,
        deviation_method=deviation_method,
        dilation_method=dilation_method,
        fill_method=fill_method
    )
    deviation = year_aggregate(deviation, np.array(years), year_aggregation_method)
    deviation = aggregate(deviation, aggregation_method)
    deviation = apply_quantiles(deviation, min_quantile, max_quantile)
    deviation = postprocess_deviation(deviation, buffer_size, dilation_method, fill_method)
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
    buffer_size, resolution, min_quantile, max_quantile, fill_method, tif_path, shape_path,
    excel_path, out_path, aggregation_method, year_aggregation_method, dilation_method, deviation_method
):
    shape_file = ogr.Open(shape_path)
    excel_file = pd.read_excel(excel_path)
    os.makedirs(out_path, exist_ok=True)
    layer = shape_file.GetLayer(0)
    for feature in layer:
        run_field(
            field=json.loads(feature.ExportToJson()),
            spatial_reference=layer.GetSpatialRef(),
            buffer_size=buffer_size,
            resolution=resolution,
            min_quantile=min_quantile,
            max_quantile=max_quantile,
            fill_method=fill_method,
            tif_path=tif_path,
            excel_file=excel_file,
            out_path=out_path,
            aggregation_method=aggregation_method,
            year_aggregation_method=year_aggregation_method,
            dilation_method=dilation_method,
            deviation_method=deviation_method
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/volume')
    parser.add_argument('--buffer-size', type=int, default=0)
    parser.add_argument('--resolution', type=float, default=10.)
    parser.add_argument('--min-quantile', type=float, default=.0)
    parser.add_argument('--max-quantile', type=float, default=1.)
    parser.add_argument('--fill-method', type=str, default='ns')
    parser.add_argument('--aggregation-method', type=str, default='mean')
    parser.add_argument('--year-aggregation-method', type=str, default='none')
    parser.add_argument('--dilation-method', type=int, default=3)
    parser.add_argument('--deviation-method', type=str, default='subtract_mean')
    options = vars(parser.parse_args())

    in_path = options['in_path']
    load_proj()
    run(
        buffer_size=options['buffer_size'],
        resolution=options['resolution'],
        min_quantile=options['min_quantile'],
        max_quantile=options['max_quantile'],
        fill_method=options['fill_method'],
        tif_path=os.path.join(in_path, 'NDVI_tif'),
        shape_path=os.path.join(in_path, 'fields.shp'),
        excel_path=os.path.join(in_path, 'NDVI_list.xls'),
        out_path=os.path.join(in_path, 'out', 'deviations'),
        aggregation_method=options['aggregation_method'],
        year_aggregation_method=options['year_aggregation_method'],
        dilation_method=options['dilation_method'],
        deviation_method=options['deviation_method']
    )
