import tempfile
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
    parser.add_argument('--image-paths', nargs='+', default=['/data/soil_line/unusable/CH'])
    parser.add_argument('--mask-paths', nargs='+', default=['/data/soil_line/bare/open_soil/173'])
    parser.add_argument('--model-path', default='/data/logs/bare/.../checkpoints/....ckpt')
    options = parser.parse_args()
    model = BaseModel.load_from_checkpoint(options.model_path)
    data_module = BaseDataModule(
        n_repeats=options.n_repeats,
        batch_size=options.batch_size,
        n_workers=options.n_workers,
        size=options.size,
        training_image_paths=None,
        training_mask_paths=None,
        validation_image_paths=options.image_paths,
        validation_mask_paths=options.mask_paths
    )
    trainer = pytorch_lightning.Trainer(
        gpus=1,
        num_sanity_val_steps=0,
        default_root_dir=tempfile.gettempdir()
    )
    trainer.fit(model, datamodule=data_module)
