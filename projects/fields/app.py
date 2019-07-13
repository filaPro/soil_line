import os
import cv2
import json
import numpy as np
import pandas as pd
from osgeo import ogr, gdal, gdalconst
from argparse import ArgumentParser


def depth(l):
    return isinstance(l, list) and (max(map(depth, l)) + 1) if l else 1


def make_mask(points, resolution):
    if depth(points) == 3:
        points = [points]
    concatenated = np.concatenate([np.concatenate(p) for p in points])
    x_min = concatenated[:, 0].min() - resolution
    y_max = concatenated[:, 1].max() + resolution
    mask_width = concatenated[:, 0].max() - x_min + resolution
    mask_height = y_max - concatenated[:, 1].min() + resolution
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


def make_circle(radius):
    return cv2.circle(
        img=np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8),
        center=(radius, radius),
        radius=radius,
        color=1,
        thickness=-1
    )


def erode(image, radius):
    return cv2.erode(image, make_circle(radius))


def dilate(deviation, full_mask, radius, fill_method, tmp_path):
    if radius == 0:
        return deviation

    if fill_method == 'ns':
        deviation = cv2.inpaint(
            src=deviation.astype(np.float32),
            inpaintMask=np.isnan(deviation).astype(np.uint8),
            inpaintRadius=1,
            flags=cv2.INPAINT_NS,
        )
        deviation[np.where(np.logical_not(full_mask))] = -1
    elif fill_method == 'm':
        deviation += 1
        deviation[np.isnan(deviation)] = 0
        for _ in range(radius * 2):
            previous = np.copy(deviation)
            deviation = cv2.dilate(deviation, kernel=make_circle(1))
            deviation[previous > 0] = previous[previous > 0]
        deviation -= 1
        deviation[np.where(np.logical_not(full_mask))] = -1
    elif fill_method == 'g':
        deviation[np.where(np.isnan(deviation))] = -1
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            utf8_path=tmp_path,
            xsize=deviation.shape[1],
            ysize=deviation.shape[0],
            bands=1,
            eType=gdal.GDT_Float32
        )
        dataset.GetRasterBand(1).WriteArray(deviation)
        del dataset
        dataset = gdal.Open(tmp_path, gdalconst.GA_Update)
        dataset.GetRasterBand(1).SetNoDataValue(-1)
        gdal.FillNodata(dataset.GetRasterBand(1), None, 2 * radius, 0)
        deviation = dataset.GetRasterBand(1).ReadAsArray()
        deviation[np.where(np.logical_not(full_mask))] = -1
    return deviation


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
    return np.nanmean(deviations, axis=0)


def apply_quantiles(deviation, min_quantile, max_quantile):
    if min_quantile > .0:
        min_threshold = np.nanquantile(deviation, min_quantile)
        deviation[deviation < min_threshold] = np.nan
    if max_quantile < 1.0:
        max_threshold = np.nanquantile(deviation, max_quantile)
        deviation[deviation > max_threshold] = np.nan
    return deviation


def save(deviation, out_path, name, spatial_reference, x_mask_min, y_mask_max, resolution):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        utf8_path=os.path.join(out_path, f'{name}.tif'),
        xsize=deviation.shape[1],
        ysize=deviation.shape[0],
        bands=1,
        eType=gdal.GDT_Float32
    )
    dataset.SetGeoTransform((x_mask_min, resolution, 0, y_mask_max, 0, resolution))
    dataset.SetProjection(spatial_reference.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(deviation)
    dataset.FlushCache()


def run_field(
        field, spatial_reference, tmp_path, buffer_size, resolution, min_quantile, max_quantile, fill_method, tif_path,
        excel_file, out_path
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
    deviation = compute_deviation(images, mask)
    deviation = apply_quantiles(deviation, min_quantile, max_quantile)
    deviation = dilate(deviation, full_mask, buffer_size, fill_method, tmp_path)
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
    excel_path, out_path
):
    shape_file = ogr.Open(shape_path)
    excel_file = pd.read_excel(excel_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
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
            out_path=out_path
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_path', type=str, default='/volume')
    parser.add_argument('--tmp_path', type=str, default='/tmp/tmp.tif')
    parser.add_argument('--buffer_size', type=int, default=0)
    parser.add_argument('--resolution', type=float, default=10.0)
    parser.add_argument('--min_quantile', type=float, default=.0)
    parser.add_argument('--max_quantile', type=float, default=1.0)
    parser.add_argument('--fill_method', type=str, default='ns')

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
        out_path=os.path.join(in_path, 'out')
    )
