import os
import torch
import numpy as np
import albumentations
import pytorch_lightning
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from dataset import BaseDataset, RepeatedDataset


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)


class DiceLoss(torch.nn.Module):
    @staticmethod
    def forward(inputs, targets):
        smooth = 1.
        inputs = torch.sigmoid(inputs)
        intersection = torch.sum((inputs * targets), dim=(1, 2))
        loss = 1 - (
            (2. * intersection + smooth) /
            (torch.sum(inputs, dim=(1, 2)) + torch.sum(targets, dim=(1, 2)) + smooth)
        )
        return torch.mean(loss)


class UNet(torch.nn.Module):
    def __init__(self, num_blocks, first_channels, image_shannels, max_width, norm_layer=torch.nn.BatchNorm2d):
        super(UNet, self).__init__()

        self.num_blocks = num_blocks
        self.image_bn = norm_layer(image_shannels)
        prev_ch = image_shannels

        # encoder
        self.encoder = torch.nn.ModuleList()
        for idx in range(num_blocks + 1):
            out_ch = min(first_channels * (2 ** idx), max_width)
            self.encoder.append(torch.nn.Sequential(
                torch.nn.MaxPool2d(2, 2, ceil_mode=True) if idx > 0 else torch.nn.Identity(),
                DownBlock(prev_ch, out_ch, norm_layer=norm_layer)
            ))
            prev_ch = out_ch

        # decoder
        self.decoder = torch.nn.ModuleList()
        for idx in reversed(range(num_blocks)):
            block_width = first_channels * (2 ** idx)
            block_ch = min(block_width, max_width)
            block_shrink = (idx > 0) and (block_width <= max_width)
            self.decoder.append(
                UpBlock(prev_ch, block_ch, shrink=block_shrink, norm_layer=norm_layer)
            )
            prev_ch = block_ch // 2 if block_shrink else block_ch

        self.final_block = DownBlock(prev_ch, 1, kernel_size=3, norm_layer=norm_layer)


    def forward(self, x, y=None):
        x = self.image_bn(x)
        encoder_output = []
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            if y is not None and i == 0:
                x = x + y
            encoder_output.append(x)
        for idx, decoder_block in enumerate(self.decoder):
            x = decoder_block(x, encoder_output[-idx - 2])
        x = self.final_block(x)
        return x[:, 0, :, :]

    def init_weights(self, pretrained=None):
        pass


class UpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, shrink=True, norm_layer=torch.nn.BatchNorm2d):
        super(UpBlock, self).__init__()
        self.upsampler = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
            torch.nn.ReLU()
        )
        self.conv3_0 = ConvBlock(2 * out_channels, out_channels, 3, norm_layer=norm_layer)
        if shrink:
            self.conv3_1 = ConvBlock(out_channels, out_channels // 2, 3, norm_layer=norm_layer)
        else:
            self.conv3_1 = ConvBlock(out_channels, out_channels, 3, norm_layer=norm_layer)

    def forward(self, x, skip):
        x = self.upsampler(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        return x


class DownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_layer=torch.nn.BatchNorm2d):
        super(DownBlock, self).__init__()
        self.down_block = torch.nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, norm_layer),
            ConvBlock(out_channels, out_channels, kernel_size, norm_layer)
        )

    def forward(self, x):
        return self.down_block(x)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_layer=torch.nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            torch.nn.ReLU(),
            norm_layer(out_channels) if norm_layer is not None else torch.nn.Identity()
        )

    def forward(self, x):
        return self.conv_block(x)


class BaseModel(pytorch_lightning.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.unet = UNet(num_blocks=5, first_channels=32, image_shannels=6, max_width=512)
        self.loss = DiceLoss()
        self.eps = 1e-7

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, _):
        image, mask = batch
        predicted = self(image)
        loss = self.loss(predicted, mask.type(predicted.dtype))
        return {
            'loss': loss,
            'log': {'train_loss': loss}
        }

    def validation_step(self, batch, _):
        image, mask = batch
        predicted = self(image)
        loss = self.loss(predicted, mask.type(predicted.dtype))
        predicted = (torch.sigmoid(predicted) > .5).type(predicted.dtype)
        intersection = torch.sum(predicted * mask, dim=(1, 2))
        union = torch.sum(mask, dim=(1, 2)) + torch.sum(predicted, dim=(1, 2)) - intersection + self.eps
        iou = torch.mean((intersection + self.eps) / union)
        return {'val_loss': loss, 'iou': iou}

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([item['val_loss'] for item in outputs], dim=0))
        iou = torch.mean(torch.stack([item['iou'] for item in outputs], dim=0))
        return {
            'val_loss': loss,
            'progress_bar': {'iou': iou, 'val_loss': loss},
            'log': {'val_loss': loss, 'iou': iou}
        }

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=.01, momentum=.9)

    def _make_dataloader(self, mask_paths, augmentation):
        return DataLoader(
            RepeatedDataset(
                BaseDataset(
                    image_path=self.hparams.image_path,
                    mask_paths=mask_paths,
                    augmentation=augmentation
                ),
                n_repeats=self.hparams.n_repeats
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.n_workers,
            worker_init_fn=lambda x: np.random.seed(x)
        )

    def train_dataloader(self):
        return self._make_dataloader(
            mask_paths=self.hparams.training_mask_paths,
            augmentation=albumentations.Compose([
                albumentations.RandomRotate90(),
                albumentations.RandomCrop(512, 512),
                ToTensorV2(),
            ])
        )

    def val_dataloader(self):
        return self._make_dataloader(
            mask_paths=self.hparams.validation_mask_paths,
            augmentation=albumentations.Compose([
                albumentations.RandomCrop(512, 512),
                ToTensorV2(),
            ])
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n-repeats', default=4)
    parser.add_argument('--batch-size', default=8)
    parser.add_argument('--n-workers', default=8)
    parser.add_argument('--image-path', default='/data/soil_line/unusable/CH')
    parser.add_argument('--mask-path', default='/data/soil_line/bare/open_soil')
    parser.add_argument('--log-path', default='/data/logs/bare/')
    options = parser.parse_args()
    mask_paths = tuple(os.path.join(options.mask_path, file_name) for file_name in os.listdir(options.mask_path))
    options.training_mask_paths = tuple(filter(lambda x: '_174' in x, mask_paths))
    options.validation_mask_paths = tuple(filter(lambda x: '_173' in x, mask_paths))
    model = BaseModel(options)
    trainer = pytorch_lightning.Trainer(
        gpus=1,
        num_sanity_val_steps=0,
        default_root_dir=options.log_path
    )
    trainer.fit(model)
