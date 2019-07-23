`app.py` parameters:
* `--in_path`, default: `/volume`
* `--tmp_path`, default: `/tmp/tmp.tif`
* `--buffer_size`, default: 0
* `--resolution`, default: 10.0
* `--min_quantile`, default: .0
* `--max_quantile`, default: 1.0
* `--fill_method`, default: `ns`
  * `ns` - Navier-Stokes inpainting with `cv2.inpaint(..., cv2.INPAINT_NS))`
  * `m` - Manhattan distance with `cv2.dilate()`
  * `g` - Euclidean distance `gdal.FillNodata()`
  * `n` - none

`classify.py` parameters:
* `--n_classes`
* `--in_path`, default: `/volume/out/deviations`
* `--method`, default: `s`
  * `s` - single
  * `m` - multiple
* `--missing_value`, default: -1.0

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
sudo docker run -ti -v /mnt/<mount_name>:/volume gdal:latest python3 app.py --buffer_size 3
sudo docker run -ti -v /mnt/<mount_name>:/volume gdal:latest python3 classify.py --n_classes 3
```
