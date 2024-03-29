import os
import torch
import geopandas
import albumentations
import pytorch_lightning
from functools import partial
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback

from dataset import BaseDataModule
from utils import generate_or_read_labels
from pytorch_model import BaseModel, pytorch_transform

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training-image-path', type=str, default='/data/soil_line/unusable/CH/174')
    parser.add_argument('--validation-image-path', type=str, default='/data/soil_line/unusable/CH/173')
    parser.add_argument('--shape-path', type=str, default='/data/soil_line/unusable/fields_v2/fields.shp')
    parser.add_argument('--excel-path', type=str, default='/data/soil_line/unusable/fields_v2/flds_all_good.xls')
    parser.add_argument('--log-path', type=str, default='/logs/unusable/')
    parser.add_argument('--pretrained-path', type=str, default='/root/.cache/torch/resnet18-f37072fd.pth',
                        help='torchvision/models/resnet.py')
    parser.add_argument('--n-training-batches', type=int, default=600)
    parser.add_argument('--n-validation-batches', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n-processes', type=int, default=16)
    parser.add_argument('--buffer-size', type=int, default=32, help='per class')
    parser.add_argument('--buffer-update-size', type=int, default=4, help='per class')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--resolution', type=float, default=30.)
    options = parser.parse_args()

    trainer = pytorch_lightning.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=12,
        limit_train_batches=options.n_training_batches,
        limit_val_batches=options.n_validation_batches,
        val_check_interval=options.n_training_batches,
        default_root_dir=options.log_path,
        num_sanity_val_steps=0,
        callbacks=[
            ModelCheckpoint(save_last=True)
        ]
    )
    fields = geopandas.read_file(options.shape_path).set_index('name')
    label_lambda = partial(
        generate_or_read_labels,
        excel_path=options.excel_path,
        fields=fields
    )
    data_module = BaseDataModule(
        training_labels=label_lambda(
            image_path=options.training_image_path,
            # uncomment for faster debug
            # label_path=os.path.join(os.path.dirname(options.excel_path), 'training.csv')
        ),
        validation_labels=label_lambda(
            image_path=options.validation_image_path,
            # uncomment for faster debug
            # label_path=os.path.join(os.path.dirname(options.excel_path), 'validation.csv')
        ),
        training_image_path=options.training_image_path,
        validation_image_path=options.validation_image_path,
        training_transform=partial(
            pytorch_transform,
            augmentation=albumentations.Compose([
                albumentations.HorizontalFlip(),
                albumentations.VerticalFlip()
            ]),
        ),
        validation_transform=pytorch_transform,
        fields=fields,
        batch_size=options.batch_size,
        n_processes=options.n_processes,
        buffer_size=options.buffer_size,
        buffer_update_size=options.buffer_update_size,
        image_size=options.image_size,
        resolution=options.resolution,
        get_current_epoch=lambda: trainer.current_epoch
    )
    model = BaseModel(pretrained=options.pretrained_path)
    trainer.fit(model, datamodule=data_module)
