import torch
import pytorch_lightning
from torch.nn import functional


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
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(256, 512, 3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(256, 512, 3),
            torch.nn.BatchNorm2d(128),
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
        logits = self(batch['image'])
        loss = self.loss(logits, batch['labels'])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch['image'])
        loss = self.loss(logits, batch['labels'])
        self.log('val_loss', loss, prog_bar=True)
        accuracy = self.accuracy(logits, batch['labels'])
        self.log('val_acc', accuracy, prog_bar=True)

    def test_step(self, barch, _):
        pass  # todo: ?

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [14, 15])
        return [optimizer], [scheduler]