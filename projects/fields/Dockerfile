FROM osgeo/gdal:ubuntu-small-latest

RUN apt-get update
RUN apt-get install -y python3-pip libopencv-dev; exit 0
RUN apt-get install -y python3-pip libopencv-dev

RUN pip3 install numpy \
    GDAL \
    pandas \
    opencv-python \
    xlrd

WORKDIR /home/soil_line/projects/fields
