import os

import torch
import pandas as pd
import geopandas
from argparse import ArgumentParser

from dataset import BaseDataModule
from utils import generate_or_read_labels
from pytorch_model import pytorch_transform
from pytorch_model_resnet import ResnetModel


def predict(model, options, test_image_path, label_path=None, device='cuda'):
    fields = geopandas.read_file(options.shape_path).set_index('name')
    labels = generate_or_read_labels(
        image_path=test_image_path,
        fields=fields,
        # uncomment for faster debug
        label_path=label_path
    )
    result = pd.DataFrame(.0, index=labels.index, columns=labels.columns)
    dataloader = BaseDataModule(
        fields=fields,
        resolution=options.resolution,
        test_labels=labels,
        test_image_path=test_image_path,
        n_processes=options.n_processes,
        image_size=options.image_size,
        test_transform=pytorch_transform,
        batch_size=options.batch_size
    ).test_dataloader()
    for batch in dataloader:
        with torch.no_grad():
            probabilities = model(batch['image'].to(device)).detach().cpu().numpy()
        for probability, field_name, base_file_name, in zip(
            probabilities, batch['field_name'], batch['base_file_name']
        ):
            result.loc[base_file_name, field_name] = probability
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image-path', type=str, default='/data/soil_line/unusable/CH/173')
    parser.add_argument('--shape-path', type=str, default='/data/soil_line/unusable/fields_v2/fields.shp')
    parser.add_argument('--model-path', type=str, default='/logs/unusable/lightning_logs/.../checkpoints/....ckpt')
    parser.add_argument('--label-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--resolution', type=float, default=30.)
    parser.add_argument('--n-processes', type=int, default=16)
    options = parser.parse_args()

    model = ResnetModel.load_from_checkpoint(options.model_path)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    result = predict(model, options, options.image_path, label_path=options.label_path, device=device)

    result_path = options.model_path + '_results/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result.to_csv(os.path.join(result_path, 'result.csv'))
