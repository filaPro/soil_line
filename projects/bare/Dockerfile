FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-get update --fix-missing
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get install -y python3-pip libopencv-dev
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y
RUN apt-get update --fix-missing
RUN apt-get install -y --no-install-recommends libgdal-dev

RUN pip install GDAL==3.0.4 \
    albumentations \
    pytorch_lightning

