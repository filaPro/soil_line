import os
import tensorflow as tf


class Writer:
    def __init__(self, path, size):
        self.path = path
        self.size = size
        self.n_examples = 0
        self.n_records = 0
        self.name = None
        self.writer = None

    def write(self, string):
        self.writer.write(string)
        self.n_examples += 1
        if self.n_examples == self.size:
            self.reset(None)

    def reset(self, name):
        print(f'examples in previous record: {self.n_examples}')
        self.close()
        if name:
            self.name = name
            self.n_records = 0
        self.n_examples = 0
        self.n_records += 1
        path = os.path.join(self.path, f'{self.name}_{str(self.n_records - 1).zfill(3)}.tfrecord')
        self.writer = tf.io.TFRecordWriter(path)
        print(f'name: {self.name}, records: {self.n_records}')

    def close(self):
        if self.writer:
            self.writer.close()
