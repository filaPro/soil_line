import os
import sys
import cv2
import numpy as np
from osgeo import gdal, gdalconst


def load_proj():
    if getattr(sys, 'frozen', False):  # if we are inside .exe
        # noinspection PyUnresolvedReferences, PyProtectedMember
        os.environ['PROJ_LIB'] = os.path.join(sys._MEIPASS, 'proj')
    elif sys.platform == 'win32':
        os.environ['PROJ_LIB'] = os.path.join(os.path.split(sys.executable)[0], 'Library', 'share', 'proj')


def depth(l_):
    return ((max(map(depth, l_)) + 1) if l_ else 1) if isinstance(l_, list) else 0


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


def dilate(deviation, radius, fill_method, tmp_path):
    if radius == 0:
        return deviation

    if fill_method == 'ns':
        deviation = cv2.inpaint(
            src=deviation.astype(np.float32),
            inpaintMask=np.isnan(deviation).astype(np.uint8),
            inpaintRadius=1,
            flags=cv2.INPAINT_NS,
        )
    elif fill_method == 'm':
        deviation += 1
        deviation[np.isnan(deviation)] = 0
        for _ in range(radius * 10):
            previous = np.copy(deviation)
            deviation = cv2.dilate(deviation, kernel=make_circle(1))
            deviation[previous > 0] = previous[previous > 0]
        deviation -= 1
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
        gdal.FillNodata(dataset.GetRasterBand(1), None, radius * 10, 0)
        deviation = dataset.GetRasterBand(1).ReadAsArray()
    else:
        raise ValueError(f'Invalid fill_method: {fill_method}')
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
    dataset.SetGeoTransform((x_mask_min, resolution, 0, y_mask_max, 0, -resolution))
    dataset.SetProjection(spatial_reference.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(deviation)
    dataset.FlushCache()


load_proj()
