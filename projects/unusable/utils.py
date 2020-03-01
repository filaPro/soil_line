import cv2
import json
import numpy as np
from osgeo import ogr

N_PROCESSES = 16


def depth(l):
    return ((max(map(depth, l)) + 1) if l else 1) if isinstance(l, list) else 0


def reshape_points(points):
    if depth(points) == 3:
        return [points]
    return points


def make_cropped_mask(points, resolution, x_min, y_max, height, width):
    points = reshape_points(points)
    mask = np.zeros((height, width), dtype=np.uint8)
    for subfield in points:
        for i, contour in enumerate(subfield):
            contour = np.array(contour)[:, :2]
            contour[:, 0] = (contour[:, 0] - x_min) // resolution
            contour[:, 1] = (y_max - contour[:, 1]) // resolution
            color = 1 if i == 0 else 0
            cv2.fillPoly(mask, np.reshape(contour, (1, len(contour), 2)).astype(np.int32), color)
    return mask


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


def read_masks(shape_path, resolution):
    shape_file = ogr.Open(shape_path)
    layer = shape_file.GetLayer(0)
    masks = {}
    for i, feature in enumerate(layer):
        field = json.loads(feature.ExportToJson())
        points = field['geometry']['coordinates']
        mask, x_mask_min, y_mask_max, _, _ = make_mask(points, resolution)
        masks[field['properties']['name']] = {
            'id': field['properties']['id'] - 1,
            'mask': mask,
            'x': x_mask_min + mask.shape[1] * resolution / 2,
            'y': y_mask_max - mask.shape[0] * resolution / 2
        }
    return masks
