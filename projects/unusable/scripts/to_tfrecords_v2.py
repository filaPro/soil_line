import os
import numpy as np
import pandas as pd
from osgeo import gdal
import tensorflow as tf
from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser

from utils import RESOLUTION, read_masks, list_channels, list_tif_files


def run(tif_path, shape_path, excel_path, out_path, n_processes):
    masks = read_masks(shape_path)
    data_frame = pd.read_excel(excel_path)
    base_file_names = list_tif_files(tif_path)
    with Pool(n_processes) as pool:
        pool.map(partial(
            run_image,
            tif_path=tif_path,
            masks=masks,
            data_frame=data_frame,
            out_path=out_path
        ), base_file_names)


def run_image(base_file_name, tif_path, masks, data_frame, out_path, resolution=RESOLUTION):
    writer = tf.io.TFRecordWriter(os.path.join(out_path, f'{base_file_name}.tfrecord'))
    feature = {}
    for i, tif_file_name in enumerate(list_channels(base_file_name)):
        tif_file = gdal.Open(os.path.join(tif_path, tif_file_name))
        x_min, _, _, y_max, _, _ = tif_file.GetGeoTransform()
        image = tif_file.GetRasterBand(1).ReadAsArray()
        x_max = x_min + resolution * image.shape[1]
        y_min = y_max - resolution * image.shape[0]
        feature[f'channels/{i}'] = tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(image)))
    feature['x_min'] = tf.train.Feature(float_list=tf.train.FloatList(value=(x_min,)))
    feature['y_max'] = tf.train.Feature(float_list=tf.train.FloatList(value=(y_max,)))
    feature['width'] =  tf.train.Feature(int64_list=tf.train.Int64List(value=(image.shape[1],)))
    feature['height'] = tf.train.Feature(int64_list=tf.train.Int64List(value=(image.shape[0],)))
    positive_names = data_frame[data_frame['NDVI_map'] == f'{base_file_name}_n.tif']['name'].values
    print(set(data_frame['name']) - set(masks.keys()))  # TODO: temporary bug in dataset
    positive_names = list(set(positive_names) & set(masks.keys()))  # TODO: temporary bug in dataset
    negative_names = []
    for name in masks.keys():
        if not name in positive_names and x_min < masks[name]['x'] < x_max and y_min < masks[name]['y'] < y_max:
            negative_names.append(name)
    feature['positive'] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=[masks[name]['id'] for name in positive_names])
    )
    feature['negative'] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=[masks[name]['id'] for name in negative_names])
    )
    writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
    writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/data')
    parser.add_argument('--out-path', type=str, default='/volume/unusable_tfrecords_v2')
    parser.add_argument('--n-processes', type=int, default=16)
    options = vars(parser.parse_args())

    run(
        tif_path=os.path.join(options['in_path'], 'CH'),
        shape_path=os.path.join(options['in_path'], 'fields.shp'),
        excel_path=os.path.join(options['in_path'], 'NDVI_list.xls'),
        out_path=options['out_path'],
        n_processes=options['n_processes']
    )

