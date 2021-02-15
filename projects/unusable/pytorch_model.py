import torch
import numpy as np
import pytorch_lightning
from torch.nn import functional


def pytorch_transform(images, label, augmentation=None):
    images['mask'] = tuple(images.values())[0].mask.astype(np.float32)
    for key, value in images.items():
        images[key] = value.data
    nir = images['nir']
    red = images['red']
    images['ndvi'] = (nir - red) / (nir + red)
    image = np.stack(tuple(images.values()), axis=-1)
    if augmentation is not None:
        image = augmentation(image=image)['image']
    image = np.moveaxis(image, -1, 0)
    return {
        'image': image,
        'label': label
    }


class BaseModel(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(8, 64, 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(256, 512, 3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(512, 512, 3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(.5),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(.5),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )
        self.loss = functional.binary_cross_entropy_with_logits
        self.accuracy = pytorch_lightning.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        probabilities = self(batch['image'])
        labels = torch.unsqueeze(batch['label'], dim=-1)
        loss = self.loss(probabilities, labels.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        probabilities = self(batch['image'])
        labels = torch.unsqueeze(batch['label'], dim=-1)
        loss = self.loss(probabilities, labels.float())
        self.log('val_loss', loss, prog_bar=True)
        accuracy = self.accuracy(probabilities, labels)
        self.log('val_acc', accuracy, prog_bar=True)

    def test_step(self, batch, _):
        pass  # todo: ?

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [14, 15])
        return [optimizer], [scheduler]
