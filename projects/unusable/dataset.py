import numpy as np
import pytorch_lightning
from torch.utils.data import IterableDataset, DataLoader

from utils_v1 import read_masked_images


class BaseDataset(IterableDataset):
    def __init__(self, image_path, fields, transform, image_size, resolution,
                 labels, buffer_size, buffer_update_size):
        super().__init__()
        self.image_path = image_path
        self.fields = fields
        self.transform = transform
        self.image_size = image_size
        self.resolution = resolution
        self.labels = labels
        self.buffer_size = buffer_size
        self.buffer_update_size = buffer_update_size
        self.n_classes = 2
        self.buffers = [list() for _ in range(self.n_classes)]
        self.base_file_names = [self.labels.index[np.any(labels == i, axis=1)]
                                for i in range(self.n_classes)]

    def _fill_buffers(self):
        while True:
            # list not full buffers
            labels = [i for i in range(self.n_classes) if len(self.buffers[i]) < self.buffer_size]
            if not len(labels):
                break

            # choice one image and several fields to read from it
            base_file_name = np.random.choice(self.base_file_names[labels[0]])
            names = []
            for label in labels:
                label_names = self.labels.loc[base_file_name].isin((label,)).index
                names += np.random.permutation(label_names)[:self.buffer_update_size].tolist()

            # read masked fields from all channels
            images = read_masked_images(
                image_path=self.image_path,
                fields=self.fields,
                base_file_name=base_file_name,
                names=names,
                image_size=self.image_size,
                resolution=self.resolution
            )

            # update buffers with read masked fields
            for name, image in images.items():
                label = self.labels.loc[base_file_name, name]
                buffer = self.buffers[label]
                buffer.append(self.transform(images=image, label=label))

    def __iter__(self):
        while True:
            self._fill_buffers()
            buffer = self.buffers[np.random.randint(self.n_classes)]
            yield buffer.pop(np.random.randint(len(buffer)))


class BaseDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self,
                 fields,
                 batch_size,
                 n_processes,
                 image_size,
                 resolution,
                 training_labels,
                 validation_labels,
                 training_image_path,
                 validation_image_path,
                 training_transform,
                 validation_transform,
                 buffer_size,
                 buffer_update_size):
        super().__init__()
        self.fields = fields
        self.batch_size=batch_size
        self.n_processes=n_processes
        self.image_size=image_size
        self.resolution=resolution
        self.training_labels=training_labels
        self.validation_labels=validation_labels
        self.training_image_path=training_image_path
        self.validation_image_path=validation_image_path
        self.training_transform = training_transform
        self.validation_transform = validation_transform
        self.buffer_size = buffer_size
        self.buffer_update_size = buffer_update_size

    def _make_dataloader(self, image_path, labels, transform):
        return DataLoader(
            BaseDataset(
                image_path=image_path,
                fields=self.fields,
                transform=transform,
                image_size=self.image_size,
                resolution=self.resolution,
                labels=labels,
                buffer_size=self.buffer_size,
                buffer_update_size=self.buffer_update_size
            ),
            batch_size=self.batch_size,
            num_workers=self.n_processes,
            worker_init_fn=lambda x: np.random.seed(x)
        )

    def train_dataloader(self):
        return self._make_dataloader(
            image_path=self.training_image_path,
            labels=self.training_labels,
            transform=self.training_transform
        )

    def val_dataloader(self):
        return self._make_dataloader(
            image_path=self.validation_image_path,
            labels=self.validation_labels,
            transform=self.validation_transform
        )
