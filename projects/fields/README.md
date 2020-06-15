`app.py` parameters:
* `--in_path`, default: `/volume`
* `--tmp_path`, default: `/tmp/tmp.tif`
* `--buffer_size`, default: 0
* `--resolution`, default: 10.
* `--min_quantile`, default: .0
* `--max_quantile`, default: 1.
* `--fill_method`, default: `ns`
  * `ns` - Navier-Stokes inpainting with `cv2.inpaint(..., cv2.INPAINT_NS))`
  * `m` - Manhattan distance with `cv2.dilate()`
  * `g` - Euclidean distance with `gdal.FillNodata()`
  * `n` - none
* `--aggregation_method`, default: `mean`
  * `min`, `max`, `mean`, `max_minus_min`
* `--dilation_method`, default: 3
  * `1`: images
  * `2`: deviations
  * `3`: final deviation
* `--deviation_method`, default: 1
  * `0`: do nothing
  * `1`: subtract mean NDVI

`classify.py` parameters:
* `--n_classes`
* `--sieve_threshold`, default: 0
* `--in_path`, default: `/volume/out/deviations`
* `--tmp_path`, default: `/tmp/tmp.tif`
* `--method`, default: `s`
  * `s` - single
  * `m` - multiple
* `--missing_value`, default: -1.

`preprocess.py` parameters:
* `in_path`, default: `/volume`
* `tmp_path`, default: `/tmp/tmp.tif`
* `--fill_method`, default: `ns`

Input files structure:
```
<in_path>
-- fields.*
-- NDVI_list.xls
-- NDVI_tif
 -- *.tif
```

Mount:
```bash
sudo mkdir /mnt/<mount_name>
sudo mount -t vboxsf <folder_name> /mnt/<mount_name>
```

Build:
```bash
cd /home/ruh/gdal/soil_line
git pull
cd projects/fields
sudo docker build --no-cache -t gdal .
```

Run:
```bash
sudo docker run -ti -v /mnt/<mount_name>:/volume gdal:latest python3 preprocess.py --fill_method g
sudo docker run -ti -v /mnt/<mount_name>:/volume gdal:latest python3 app.py --buffer_size 3
sudo docker run -ti -v /mnt/<mount_name>:/volume gdal:latest python3 classify.py --n_classes 3
```

-------


Make exe:
```bash
pyinstaller many.spec
```

Exe usage example:
```bash
preprocess.exe --in_path D:\SoilLineData --tmp_path D:\SoilLineData\tmp\tmp.tif
app.exe --in_path D:\SoilLineData --tmp_path D:\SoilLineData\tmp\tmp.tif
classify.exe --in_path D:\SoilLineData\out\tif --tmp_path D:\SoilLineData\tmp\tmp.tif --n_classes 3
```