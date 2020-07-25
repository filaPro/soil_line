import os
import json
import tempfile
import numpy as np
from osgeo import ogr, gdal
from shutil import copyfile
from argparse import ArgumentParser

from lib import make_cropped_mask, dilate, save, load_proj


def get_file_names(in_path):
    file_names = []
    for file_name in os.listdir(in_path):
        if file_name.endswith('.shp') and not file_name.startswith('fields'):
            file_names.append(file_name[:-4])
    return file_names


def swap(first_path, second_path):
    tmp_path = os.path.join(tempfile.gettempdir(), 'tmp.tif')
    copyfile(first_path, tmp_path)
    copyfile(second_path, first_path)
    copyfile(tmp_path, second_path)


def run_file(shape_file, tif_path, name, fill_method, out_path):
    tif_file = gdal.Open(os.path.join(tif_path, f'{name}.tif'))
    x_min, resolution, _, y_max, _, _ = tif_file.GetGeoTransform()
    height, width = tif_file.GetRasterBand(1).ReadAsArray().shape
    mask = np.zeros((height, width), dtype=np.uint8)
    layer = shape_file.GetLayer(0)
    for feature in layer:
        field = json.loads(feature.ExportToJson())
        points = field['geometry']['coordinates']
        mask = np.logical_or(mask, make_cropped_mask(points, resolution, x_min, y_max, height, width))
    image = tif_file.GetRasterBand(1).ReadAsArray()
    image[mask == 1] = np.nan
    image = dilate(image, 10, fill_method)
    save(image, out_path, name, tif_file.GetSpatialRef(), x_min, y_max, resolution)
    swap(
        first_path=os.path.join(tif_path, f'{name}.tif'),
        second_path=os.path.join(out_path, f'{name}.tif')
    )


def run(in_path, fill_method, tif_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    for file_name in os.listdir(in_path):
        if file_name.endswith('.shp') and not file_name.startswith('fields'):
            print(file_name)
            shape_file = ogr.Open(os.path.join(in_path, file_name))
            name = file_name[:-4]
            run_file(shape_file, tif_path, name, fill_method, out_path)


if __name__ == '__main__':
    load_proj()

    parser = ArgumentParser()
    parser.add_argument('--in_path', type=str, default='/volume')
    parser.add_argument('--fill_method', type=str, default='ns')

    options = vars(parser.parse_args())
    run(
        in_path=options['in_path'],
        fill_method=options['fill_method'],
        tif_path=os.path.join(options['in_path'], 'NDVI_tif'),
        out_path=os.path.join(options['in_path'], 'out', 'tif')
    )
