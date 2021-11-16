import torch
import numpy as np
from skimage.transform import resize
import pytorch_lightning
from torch.nn import functional
from torch.utils.data.dataloader import default_collate
from torchmetrics import Accuracy


def pytorch_transform(images, label, field_name, base_file_name, augmentation=None, apply_mask=True, resize_hw=None):
    images['mask'] = 1 - tuple(images.values())[0].mask.astype(np.float32)

    for key, value in images.items():
        images[key] = value.data
    nir = images['nir']
    red = images['red']
    images['ndvi'] = (nir - red) / (nir + red + .0001)

    if apply_mask:
        for key in images.keys() - {'mask'}:
            images[key] *= images['mask']

    image = np.stack(tuple(images.values()), axis=-1)

    if resize_hw:
        image = resize(image, [resize_hw, resize_hw])

    if augmentation is not None:
        image = augmentation(image=image)['image']
    image = np.moveaxis(image, -1, 0)
    return {
        'image': image,
        'label': label,
        'field_name': field_name,
        'base_file_name': base_file_name
    }


class BaseModel(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, (3, 3)),
            # torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(16, 32, (3, 3)),
            # torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(32, 64, (3, 3)),
            torch.nn.LayerNorm([64, 28, 28]),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(64, 64, (3, 3)),
            torch.nn.LayerNorm([64, 12, 12]),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(64, 64, (3, 3)),
            torch.nn.LayerNorm([64, 4, 4]),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            # torch.nn.Linear(64, 64),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(.5),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        self.loss = functional.binary_cross_entropy
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        probabilities = self(batch['image'])
        labels = torch.unsqueeze(batch['label'], dim=-1)
        loss = self.loss(probabilities, labels.float())
        self.log('train_loss', loss.mean())

        pred = probabilities > 0.5
        fp = sum(pred * (1-labels))
        fn = sum((~pred) * labels)
        tp = sum(pred * labels)
        return {'loss': loss, 'fp': fp, 'fn': fn, 'tp': tp}

    def training_epoch_end(self, outputs):
        res = default_collate(outputs)
        fp = res['fp'].sum()
        fn = res['fn'].sum()
        tp = res['tp'].sum()
        self.log('train_recall', tp/(tp + fn))
        self.log('train_precision', tp/(tp + fp))
        print(f'train: recall={tp/(tp + fn)}, precision={tp/(tp + fp)}')

    def validation_step(self, batch, _):
        probabilities = self(batch['image'])
        labels = torch.unsqueeze(batch['label'], dim=-1)
        loss = self.loss(probabilities, labels.float())
        self.log('val_loss', loss, prog_bar=True)
        accuracy = self.accuracy(probabilities, labels)
        self.log('val_acc', accuracy, prog_bar=True)

        pred = probabilities > 0.5
        fp = sum(pred * (1-labels))
        fn = sum((~pred) * labels)
        tp = sum(pred * labels)
        return {'fp': fp, 'fn': fn, 'tp': tp}

    def validation_epoch_end(self, outputs):
        res = default_collate(outputs)
        fp = res['fp'].sum()
        fn = res['fn'].sum()
        tp = res['tp'].sum()
        self.log('val_recall', tp/(tp + fn))
        self.log('val_precision', tp/(tp + fp))
        print(f'validation: recall={tp/(tp + fn)}, precision={tp/(tp + fp)}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [8, 11])
        return [optimizer], [scheduler]


def log_weights(*args):
    model = args[1]

    if model.global_step % 10 == 0:
        for i, layer in enumerate(list(model.model.modules())[1:]):
            for j, p in enumerate(layer.parameters()):
                c_name = f'layer_{i}_{layer.__repr__()}_param_{j}'
                model.log(c_name + '_mean', p.mean())
                model.log(c_name + '_disp', p.std())
                model.log(c_name + '_grad_mean', p.grad.mean())
                model.log(c_name + '_grad_disp', p.grad.std())
