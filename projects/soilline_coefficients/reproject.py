import os
import tempfile

import gdal


def reproject_vrt(src_proj, src_gt, src_size, dst_proj):
    driver = gdal.GetDriverByName('VRT')
    src_ds = driver.Create('', src_size[0], src_size[1])

    src_ds.SetGeoTransform(src_gt)
    src_ds.SetProjection(src_proj)

    dst_ds = gdal.AutoCreateWarpedVRT(
        src_ds,
        src_proj,
        dst_proj)

    return dst_ds.GetGeoTransform(), dst_ds.RasterXSize, dst_ds.RasterYSize


def open_with_reproject(path, projection, reproject_path=tempfile.gettempdir()):
    name = os.path.split(path)[-1]
    if not os.path.exists(reproject_path):
        os.makedirs(reproject_path)

    src_ds = gdal.Open(path)
    if src_ds is None:
        return

    if src_ds.GetProjection() == projection:
        return src_ds

    dst_gt, dst_width, dst_height = reproject_vrt(src_ds.GetProjection(), src_ds.GetGeoTransform(),
                                                  (src_ds.RasterXSize, src_ds.RasterYSize), projection)

    dst_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(reproject_path, name),
                                                  dst_width, dst_height, 1, gdal.GDT_Float32)

    dst_ds.SetGeoTransform(dst_gt)
    dst_ds.SetProjection(projection)

    gdal.ReprojectImage(src_ds, dst_ds, None, None, gdal.GRA_NearestNeighbour)

    dst_ds.FlushCache()

    return dst_ds
