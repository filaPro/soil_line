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
sudo docker build -t gdal .
```

Run:
```bash
sudo docker run -ti -v <mount_name>:/volume gdal:latest python3 app.py --in_path /volume --buffer_size 3
```

