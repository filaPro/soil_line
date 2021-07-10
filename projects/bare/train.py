import pytorch_lightning
from argparse import ArgumentParser

from dataset import BaseDataModule
from model import BaseModel


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--n-repeats', default=4)
    parser.add_argument('--batch-size', default=8)
    parser.add_argument('--n-workers', default=8)
    parser.add_argument('--size', default=512)
    parser.add_argument('--training-image-paths', nargs='+', default=['/data/soil_line/unusable/CH'])
    parser.add_argument('--training-mask-paths', nargs='+', default=['/data/soil_line/bare/rz/open_soil/174'])
    parser.add_argument('--validation-image-paths', nargs='+', default=['/data/soil_line/unusable/CH'])
    parser.add_argument('--validation-mask-paths', nargs='+', default=['/data/soil_line/bare/rz/open_soil/173'])
    parser.add_argument('--log-path', default='/data/logs/bare/')
    options = parser.parse_args()
    model = BaseModel()
    data_module = BaseDataModule(
        n_repeats=options.n_repeats,
        batch_size=options.batch_size,
        n_workers=options.n_workers,
        size=options.size,
        training_image_paths=options.training_image_paths,
        training_mask_paths=options.training_mask_paths,
        validation_image_paths=options.validation_image_paths,
        validation_mask_paths=options.validation_mask_paths
    )
    trainer = pytorch_lightning.Trainer(
        gpus=1,
        max_epochs=12,
        default_root_dir=options.log_path
    )
    trainer.fit(model, datamodule=data_module)
