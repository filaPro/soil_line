FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN apt-get update
RUN apt-get install -y python3-rasterio
RUN pip install rasterio==1.1.3 \
    geopandas==0.9.0 \
    pandas==1.3.4 \
    albumentations==1.1.0 \
    pytorch-lightning==1.5.1 \
    xlrd==2.0.1 \
    scikit-learn==1.0.1 \
    catboost==1.0.3 \
    matplotlib==3.4.3 \
    torchmetrics==0.6.0 \
    jupyter
