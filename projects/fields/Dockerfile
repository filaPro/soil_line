FROM osgeo/gdal:ubuntu-small-latest

RUN apt-get update \
    && apt-get install -y python3-pip git libsm6 libxext6 libxrender-dev wget
RUN cd /home \
    && mkdir soil_line \
    && wget -q https://github.com/filapro/soil_line/archive/master.tar.gz -O - \
        | tar xz -C soil_line --strip-components=1 \
    && cd /home/soil_line/projects/fields \
    && pip3 install -r requirements.txt

WORKDIR /home/soil_line/projects/fields
# CMD python3 app.py