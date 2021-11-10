import torch
import logging
import numpy as np
import pytorch_lightning
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataloader import default_collate

from utils import read_masked_images


def collate_test_dataloader(batch):
    keys = set().union(*batch)
    res = {key: default_collate([d[key] for d in batch if key in d]) for key in keys}
    if '_worker_stopped' in res:
        worker_id = res.pop('_worker_stopped')[0].item()
        if not res:
            logging.info(f'Process {worker_id} finished')
            raise StopIteration
    return res


class TestDataset(IterableDataset):
    def __init__(self, image_path, fields, transform, image_size, resolution, labels):
        self.image_path = image_path
        self.fields = fields
        self.transform = transform
        self.image_size = image_size
        self.resolution = resolution
        self.labels = labels

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            begin = 0
            end = len(self.labels)
            n_workers = 1
            worker_id = 0
        else:
            n_workers = worker_info.num_workers
            worker_id = worker_info.id
            images_per_worker = int(np.ceil(len(self.labels) / n_workers))
            begin = images_per_worker * worker_id
            end = min(begin + images_per_worker, len(self.labels))
        labels = self.labels.iloc[begin: end]

        for i, base_file_name in enumerate(labels.index):
            logging.info(f'process: {worker_id}/{n_workers}; '
                         f'generate features for {base_file_name} {i}/{len(labels.index)}')

            names = labels.columns[labels.loc[base_file_name].isin(tuple(range(2)))]
            images = read_masked_images(
                image_path=self.image_path,
                fields=self.fields,
                base_file_name=base_file_name,
                names=names,
                image_size=self.image_size,
                resolution=self.resolution
            )
            for name, image in images.items():
                label = labels.loc[base_file_name][name]
                yield self.transform(
                    images=image,
                    label=label,
                    field_name=name,
                    base_file_name=base_file_name
                )

        # Preventing StopIteration to wait all processes.
        if worker_info is not None:
            while True:
                yield {'_worker_stopped': worker_id}


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
                label_names = self.labels.columns[self.labels.loc[base_file_name].isin((label,))]
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
                buffer.append(self.transform(
                    images=image,
                    label=label,
                    field_name=name,
                    base_file_name=base_file_name
                ))

    def __iter__(self):
        while True:
            self._fill_buffers()
            buffer = self.buffers[np.random.randint(self.n_classes)]
            index = np.random.randint(len(buffer))
            yield buffer.pop(index)


class BaseDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self,
                 fields,
                 n_processes,
                 image_size,
                 resolution,
                 batch_size,
                 training_labels=None,
                 validation_labels=None,
                 test_labels=None,
                 training_image_path=None,
                 validation_image_path=None,
                 test_image_path=None,
                 training_transform=None,
                 validation_transform=None,
                 test_transform=None,
                 buffer_size=None,
                 buffer_update_size=None,
                 get_current_epoch=None):
        super().__init__()
        self.fields = fields
        self.n_processes = n_processes
        self.image_size = image_size
        self.resolution = resolution
        self.batch_size = batch_size
        self.training_labels = training_labels
        self.validation_labels = validation_labels
        self.test_labels = test_labels
        self.training_image_path = training_image_path
        self.validation_image_path = validation_image_path
        self.test_image_path = test_image_path
        self.training_transform = training_transform
        self.validation_transform = validation_transform
        self.test_transform = test_transform
        self.buffer_size = buffer_size
        self.buffer_update_size = buffer_update_size
        self.get_current_epoch = get_current_epoch or (lambda: 0)  # for different random seed in different epochs

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
            worker_init_fn=lambda x: np.random.seed(x + self.get_current_epoch() * 100)
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

    def test_dataloader(self):
        return DataLoader(
            TestDataset(
                image_path=self.test_image_path,
                fields=self.fields,
                transform=self.test_transform,
                image_size=self.image_size,
                resolution=self.resolution,
                labels=self.test_labels
            ),
            batch_size=self.batch_size,
            num_workers=self.n_processes,
            collate_fn=collate_test_dataloader
        )
