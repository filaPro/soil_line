FROM osgeo/gdal:ubuntu-small-latest

RUN apt-get update --fix-missing
RUN apt-get install -y libopencv-dev python3-pip wget
RUN pip3 install opencv-python \
    pandas \
    xlrd

RUN cd /home \
    && mkdir soil_line \
    && wget -q https://github.com/filapro/soil_line/archive/master.tar.gz -O - \
        | tar xz -C soil_line --strip-components=1
WORKDIR /home/soil_line/projects/fields
