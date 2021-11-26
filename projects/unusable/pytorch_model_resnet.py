import pytorch_lightning
import torch
from pytorch_lightning.metrics import Accuracy
from torch.nn import functional
from torch.fx import symbolic_trace
from torch.utils.data.dataloader import default_collate
from torchvision.models import resnet18


class ResnetModel(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        rn = resnet18(pretrained=True)
        rn = symbolic_trace(rn)
        conv1_old = list(rn.graph.nodes)[1]
        conv1_new = torch.nn.Conv2d(8, 64, (3, 3))
        rn.add_module('my_conv1', conv1_new)
        with rn.graph.inserting_after(conv1_old):
            new_node = rn.graph.call_module('my_conv1', conv1_old.args, conv1_old.kwargs)
            conv1_old.replace_all_uses_with(new_node)
        rn.graph.erase_node(conv1_old)
        rn.recompile()

        self.model = torch.nn.Sequential(
            rn,
            torch.nn.Linear(1000, 1),
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
