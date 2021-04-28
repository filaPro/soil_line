import os
import torch
import pandas
import geopandas
from argparse import ArgumentParser

from dataset import BaseDataModule
from utils_v1 import generate_or_read_labels
from pytorch_model import BaseModel, pytorch_transform


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image-path', type=str, default='/data/soil_line/unusable/CH/173')
    parser.add_argument('--shape-path', type=str, default='/data/soil_line/unusablefields.shp')
    parser.add_argument('--model-path', type=str, default='/data/logs/....ckpt')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--resolution', type=float, default=30.)
    parser.add_argument('--n-processes', type=float, default=16)
    options = parser.parse_args()

    model = BaseModel.load_from_checkpoint(options.model_path)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    fields = geopandas.read_file(options.shape_path).set_index('name')
    labels = generate_or_read_labels(
        image_path=options.image_path,
        fields=fields,
        # uncomment for faster debug
        # label_path=os.path.join(os.path.dirname(options.shape_path), 'validation.csv')
    )
    result = pandas.DataFrame(.0, index=labels.index, columns=labels.columns)
    dataloader = BaseDataModule(
        fields=fields,
        resolution=options.resolution,
        test_labels=labels,
        test_image_path=options.image_path,
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
    result.to_csv(os.path.join(os.path.dirname(options.model_path), 'result.csv'))
