Script parameters:
* `--in_path`, default: `/volume`
* `--tmp_path`, default: `/tmp/tmp.tif`
* `--buffer_size`, default: 0
* `--resolution`, default: 10.0

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
```
