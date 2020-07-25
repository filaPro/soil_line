import os
import numpy as np
from osgeo import gdal
from argparse import ArgumentParser

from lib import load_proj


def compute_quantiles(images, n_classes, missing_value):
    data = np.empty(0)
    for image in images:
        data = np.concatenate((data, image[image != missing_value]))
    if data.size == 0:
        return np.ones(n_classes + 1) * missing_value
    return np.quantile(data, np.arange(n_classes + 1) / n_classes)


def compute_method_quantiles(method, images, n_classes, missing_value):
    if method == 's':
        return [compute_quantiles([image], n_classes, missing_value) for image in images]
    else:  # method == 'm'
        return [compute_quantiles(images, n_classes, missing_value)] * len(images)


def read_images(path):
    print('read: begin')
    result = []
    for file_name in os.listdir(path):
        dataset = gdal.Open(os.path.join(path, file_name))
        result.append({
            'file_name': file_name,
            'image': dataset.GetRasterBand(1).ReadAsArray(),
            'transform': dataset.GetGeoTransform(),
            'projection': dataset.GetSpatialRef().ExportToWkt()
        })
    print('read: end')
    return result


def classify(image, quantiles, missing_value):
    result = np.ones(image.shape) * missing_value
    for i in range(1, len(quantiles)):
        result[np.where(np.all(np.stack((
            image != missing_value,
            image <= quantiles[i],
            image >= quantiles[i - 1]
        ), axis=0), axis=0))] = i
    return result


def sieve_filter(image, missing_value, sieve_threshold):
    driver = gdal.GetDriverByName('MEM')
    dataset = driver.Create(
        utf8_path='',
        xsize=image.shape[1],
        ysize=image.shape[0],
        bands=3,
        eType=gdal.GDT_Int16
    )
    dataset.GetRasterBand(1).WriteArray(image)
    dataset.GetRasterBand(2).WriteArray(image != missing_value)
    gdal.SieveFilter(
        srcBand=dataset.GetRasterBand(1),
        maskBand=dataset.GetRasterBand(2),
        dstBand=dataset.GetRasterBand(3),
        threshold=sieve_threshold
    )
    return dataset.GetRasterBand(3).ReadAsArray()


def save(out_path, file_name, image, transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        utf8_path=os.path.join(out_path, file_name),
        xsize=image.shape[1],
        ysize=image.shape[0],
        bands=1,
        eType=gdal.GDT_Int16
    )
    dataset.SetGeoTransform(transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image)
    dataset.FlushCache()


def run(n_classes, sieve_threshold, in_path, out_path, method, missing_value):
    images = read_images(in_path)
    os.makedirs(out_path, exist_ok=True)
    quantiles = compute_method_quantiles(
        method=method, 
        images=[d['image'] for d in images],
        n_classes=n_classes,
        missing_value=missing_value
    )
    for quantile, data in zip(quantiles, images):
        image = classify(data['image'], quantile, missing_value)
        file_name = data['file_name']
        print(f'classify: {file_name}, quantiles: {quantile}')
        image = sieve_filter(image, missing_value, sieve_threshold)
        save(
            out_path=out_path, 
            file_name=file_name,
            image=image,
            transform=data['transform'],
            projection=data['projection']
        )


if __name__ == '__main__':
    load_proj()

    parser = ArgumentParser()
    parser.add_argument('--n_classes', type=int, required=True)
    parser.add_argument('--sieve_threshold', type=int, default=0)
    parser.add_argument('--in_path', type=str, default='/volume/out/deviations')
    parser.add_argument('--method', type=str, default='s')
    parser.add_argument('--missing_value', type=float, default=-1.)
    options = vars(parser.parse_args())
    run(
        n_classes=options['n_classes'],
        sieve_threshold=options['sieve_threshold'],
        in_path=options['in_path'],
        out_path=os.path.join(os.path.dirname(options['in_path']), 'classes'),
        method=options['method'],
        missing_value=options['missing_value']
    )
