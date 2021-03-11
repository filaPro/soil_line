FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-get update --fix-missing
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get install -y python3-pip libopencv-dev
RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y
RUN apt-get update --fix-missing
RUN apt-get install -y --no-install-recommends libgdal-dev

RUN pip install GDAL==3.0.4 \
    opencv-python \
    catboost \
    pandas \
    albumentations \
    scikit-learn \
    xlrd

