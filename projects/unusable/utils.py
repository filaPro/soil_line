import os
import pandas
import logging
import rasterio
import numpy as np
import rasterio.mask
from rasterio.vrt import WarpedVRT
from collections import defaultdict

logging.getLogger().setLevel(logging.INFO)


def list_channels(base_file_name):
    channels = {
        'blue': ['01', '02', '02'],
        'green': ['02', '03', '03'],
        'red': ['03', '04', '04'],
        'nir': ['04', '05', '08'],
        'swir1': ['05', '06', '11'],
        'swir2': ['07', '07', '12']
    }
    channel_shift = {
        'LT04': 0,
        'LT05': 0,
        'LE07': 0,
        'LC08': 1,
        'S2AB': 2
    }[base_file_name.split('_')[2]]
    return {channel: f'{base_file_name}_{channel}_{channels[channel][channel_shift]}.tif' for channel in channels}


def list_tif_files(path):
    return sorted(set('_'.join(file_name.split('_')[:4]) for file_name in os.listdir(path) if '.tif' in file_name))


def generate_or_read_labels(image_path, fields, excel_path=None, label_path=None):
    # do nothing if labels.csv already exists
    if label_path is not None and os.path.exists(label_path):
        return pandas.read_csv(label_path, index_col=0)

    base_file_names = list_tif_files(image_path)
    labels = pandas.DataFrame(0, index=base_file_names, columns=fields.index, dtype=np.uint8)
    # mark not intersecting fields
    for i, base_file_name in enumerate(base_file_names):
        logging.info(f'generating labels for {label_path} {i}/{len(base_file_names)}')

        file_name = tuple(list_channels(base_file_name).values())[0]
        path = os.path.join(image_path, file_name)
        with rasterio.open(path) as reader:
            centers = fields.to_crs(reader.crs).geometry.centroid
            for center, name in zip(centers, labels.columns):
                y, x = reader.index(center.x, center.y)
                if not 0 <= x < reader.width or not 0 <= y < reader.height:
                    labels.loc[base_file_name, name] = 2

    # mark positive fields
    if excel_path is not None:
        excel_file = pandas.read_excel(excel_path)
        # todo: .xls -> .csv and remove apply
        excel_file['NDVI_map'] = excel_file['NDVI_map'].apply(lambda x: x[:22])
        for _, row in excel_file.iterrows():
            if row['NDVI_map'] in labels.index and row['name'] in labels.columns:
                assert labels.loc[row['NDVI_map'], row['name']] != 2
                labels.loc[row['NDVI_map'], row['name']] = 1

    if label_path is not None:
        labels.to_csv(label_path)
    return labels


def mask(path, fields, size, resolution):
    images = []
    with rasterio.open(path) as reader:
        if reader.transform.a != resolution:
            assert reader.transform.a > 0
            scale = reader.transform.a / resolution
            reader = WarpedVRT(reader,
                               transform=reader.transform * reader.transform.scale(1 / scale),
                               height=reader.height * scale,
                               width=reader.width * scale)
        with reader:
            for field in fields.to_crs(reader.crs).geometry:
                pad, pad_width = (False, 0) if size is None else (True, size // 2)
                masked_image, transform = rasterio.mask.mask(
                    reader, [field], crop=True, pad=pad, filled=False, pad_width=pad_width)
                masked_image = masked_image[0]

                # pad masked image if needed and crop to (size, size)
                if size is not None:
                    y_center, x_center = reader.index(field.centroid.x, field.centroid.y)
                    y_min, x_min = reader.index(transform.c, transform.f)
                    assert 0 <= y_min <= y_center < reader.height
                    assert 0 <= x_min <= x_center < reader.width
                    y = y_center - y_min
                    x = x_center - x_min
                    height, width = masked_image.shape
                    pad_left = max(0, size // 2 - x)
                    pad_top = max(0, size // 2 - y)
                    pad_right = max(0, x + size // 2 - width)
                    pad_bottom = max(0, y + size // 2 - height)
                    if any((pad_left, pad_top, pad_right, pad_bottom)):
                        pad = ((pad_top, pad_bottom), (pad_left, pad_right))
                        image = np.pad(masked_image.data, pad)
                        mask = np.pad(masked_image.mask, pad, constant_values=True)
                        masked_image = np.ma.masked_array(image, mask)
                    left = x + pad_left - size // 2
                    right = left + size
                    top = y + pad_top - size // 2
                    bottom = top + size
                    masked_image = masked_image[top: bottom, left: right]
                    assert masked_image.shape == (size, size)
                images.append(masked_image)
    return images


def read_masked_images(image_path, fields, base_file_name, names, image_size, resolution):
    images = defaultdict(dict)
    for channel, file_name in list_channels(base_file_name).items():
        image_list = mask(
            path=os.path.join(image_path, file_name),
            fields=fields.loc[names],
            size=image_size,
            resolution=resolution
        )
        for name, image in zip(names, image_list):
            images[name][channel] = image
    return images
