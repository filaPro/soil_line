import torch
import pytorch_lightning


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
    def __init__(self):
        super().__init__()
        self.unet = UNet(num_blocks=5, first_channels=32, image_shannels=6, max_width=512)
        self.loss = DiceLoss()
        self.eps = 1e-7

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, _):
        image, mask = batch
        predicted = self(image)
        return self.loss(predicted, mask.type(predicted.dtype))

    def validation_step(self, batch, _):
        image, mask = batch
        predicted = self(image)
        loss = self.loss(predicted, mask.type(predicted.dtype))
        predicted = (torch.sigmoid(predicted) > .5).type(predicted.dtype)
        intersection = torch.sum(predicted * mask, dim=(1, 2))
        union = torch.sum(mask, dim=(1, 2)) + torch.sum(predicted, dim=(1, 2)) - intersection + self.eps
        iou = torch.mean((intersection + self.eps) / union)
        return {'val_loss': loss, 'iou': iou}

    def test_step(self, batch, _):
        return self.validation_step(batch, _)

    def training_epoch_end(self, outputs):
        self.log('train_loss', torch.mean(torch.stack([item['loss'] for item in outputs])), prog_bar=True)

    def validation_epoch_end(self, outputs):
        self.log('val_loss', torch.mean(torch.stack([item['val_loss'] for item in outputs])), prog_bar=True)
        self.log('val_iou', torch.mean(torch.stack([item['iou'] for item in outputs])), prog_bar=True)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [14, 15])
        return [optimizer], [scheduler]
