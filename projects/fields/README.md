```
<in_path>
-- fields.*
-- NDVI_list.xls
-- NDVI_tif
 -- *.tif
```

Mount:
sudo mkdir /mnt/<mount_name>
sudo mount -t vboxsf <folder_name> /mnt/<mount_name>

Build:
cd /home/ruh/gdal/soil_line
git pull
cd projects/fields
sudo docker build -t gdal .

Run:
sudo docker run -ti -v <mount_name>:/volume gdal:latest python3 app.py --in_path /volume --buffer_size 3