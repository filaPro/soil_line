`app.py` parameters:
* `--in-path`, default: `/volume`
* `--buffer-size`, default: 0
* `--resolution`, default: 10.
* `--min-quantile`, default: .0
* `--max-quantile`, default: 1.
* `--fill-method`, default: `ns`
  * `ns` - Navier-Stokes inpainting with `cv2.inpaint(..., cv2.INPAINT_NS))`
  * `m` - Manhattan distance with `cv2.dilate()`
  * `g` - Euclidean distance with `gdal.FillNodata()`
  * `n` - none
* `--aggregation-method`, default: `mean`
  * `min`, `max`, `mean`, `max_minus_min`, `median`, `quantile_0.4`
* `--year-aggregation-method`, default: `none`
  * `none`: do nothing
  * `min`, `max`, `mean`, `max_minus_min`, `median`: aggregate deviations with same year
* `--dilation-method`, default: 3
  * `1`: images
  * `2`: deviations
  * `3`: final deviation
* `--deviation-method`, default: 1
  * `raw`: do nothing
  * `subtract_mean`: subtract mean NDVI
  * `cdf`: calculate quantile of each pixel value

`classify.py` parameters:
* `--n-classes`
* `--sieve-threshold`, default: 0
* `--in-path`, default: `/volume/out/deviations`
* `--method`, default: `s`
  * `s` - single
  * `m` - multiple
* `--missing-value`, default: -1.

`preprocess.py` parameters:
* `--in-path`, default: `/volume`
* `--fill-method`, default: `ns`

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
sudo docker run -ti -v /mnt/<mount_name>:/volume gdal:latest python3 preprocess.py --fill-method g
sudo docker run -ti -v /mnt/<mount_name>:/volume gdal:latest python3 app.py --buffer-size 3
sudo docker run -ti -v /mnt/<mount_name>:/volume gdal:latest python3 classify.py --n-classes 3
```

-------


Make exe:
```bash
pyinstaller runner.spec
```

Exe usage example:
```bash
preprocess.exe --in-path D:\SoilLineData
app.exe --in-path D:\SoilLineData
classify.exe --in-path D:\SoilLineData\out\tif --n_classes 3
```