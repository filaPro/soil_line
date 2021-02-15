import os
import torch
import pandas
import logging
import geopandas
import numpy as np
from argparse import ArgumentParser

from utils_v1 import generate_or_read_labels, read_masked_images
from pytorch_model import BaseModel, pytorch_transform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image-path', type=str, default='/data/unusable/CH/173')
    parser.add_argument('--shape-path', type=str, default='/data/unusable/fields.shp')
    parser.add_argument('--model-path', type=str, default='/data/logs/unusable/lightning_logs/version_40/checkpoints/epoch=0-step=499.ckpt')  # todo: ..., ...
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--resolution', type=float, default=30.)
    options = parser.parse_args()

    model = BaseModel.load_from_checkpoint(options.model_path)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    fields = geopandas.read_file(options.shape_path).set_index('name')
    labels = generate_or_read_labels(
        image_path=options.image_path,
        fields=fields,
        # todo: remove label_path
        label_path=os.path.join(os.path.dirname(options.shape_path), 'validation.csv')
    )
    result = pandas.DataFrame(.0, index=labels.index, columns=labels.columns)
    for i, base_file_name in enumerate(labels.index[:2]):
        logging.info(f'predicting for {base_file_name} {i}/{len(labels.index)}')

        names = labels.loc[base_file_name].isin(tuple(range(2))).index
        images = read_masked_images(
            image_path=options.image_path,
            fields=fields,
            base_file_name=base_file_name,
            names=names,
            image_size=options.image_size,
            resolution=options.resolution
        )

        for j in range(int(np.ceil(len(names) / options.batch_size))):
            begin = options.batch_size * j
            end = options.batch_size * (j + 1)
            batch_images, batch_names = [], []
            for name in tuple(images.keys())[begin:end]:
                batch_images.append(pytorch_transform(images[name], 0)['image'])
                batch_names.append(name)
            batch = torch.from_numpy(np.stack(tuple(batch_images))).to(device)
            with torch.no_grad():
                probabilities = model(batch).detach().cpu().numpy()
            for name, probability in zip(batch_names, probabilities):
                result.loc[base_file_name, name] = probability
    result.to_csv(os.path.join(os.path.dirname(options.model_path), 'result.csv'))




