"""

Contains data generators for training and prediction
tasks.

Author: Simon Thomas
Date: 30th October 2020

Requirements (available by pip):
- skimage
- mrcfile
- numpy
"""
import mrcfile
import numpy as np

from skimage import filters
from skimage import transform


class _Generator:
    """
    A hidden class for creating separate generators for training, validation and testing splits. This
    enables the same MRC files to be referenced from different generators that have pointers to a
    shared parent.
    """
    def __init__(self, indices, parent):
        self.indices = indices
        self.parent = parent
        self.n = len(self.indices)

    def __getitem__(self, i):
        return self.parent.__getitem__(i)


class DataGenerator:
    """
    Main generator class which controls the construction of sub-generators
    for training, validation and test phases.
    """
    def __init__(self,
                 pos_files: list,
                 neg_files: list,
                 batch_size: int,
                 shuffle: bool = True,
                 data_augmentation: bool = True,
                 split: float = 0.7,
                 img_dim: int = 28
                 ):
        """
        :param pos_files: a list of mrcs files for the positive class
        :param neg_files: a list of mrcs files for the negative class
        :param batch_size: integer designating the batch size of the generator
        :param shuffle: bool for shuffling the stacks
        :param data_augmentation: bool for performing data augmentation (flip & rotate)
        :param split: float between 0-1 (inclusive) for training proportion (val and test are equal)
        :param img_dim: int designating the size of the images (img_dim x img_dim)
        """
        self.pos_files = pos_files
        self.neg_files = neg_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.split = split
        self.img_dim = img_dim

        # Configure training indices
        self.n_pos = len(self.pos_files)
        self.n_neg = len(self.neg_files)

        # Open mrc files
        self.pos_mrc = [
            mrcfile.open(path) for path in self.pos_files
        ]
        self.neg_mrc = [
            mrcfile.open(path) for path in self.neg_files
        ]

        # Create indices and file/label pointers
        self.indices = []
        self.mrcs = {}
        self.label = {}
        for i, mrc in enumerate(self.pos_mrc):
            self.indices.extend([ (i, p) for p in range(mrc.data.shape[0])])
            self.mrcs[i] = mrc
            self.label[i] = 1.
        for i, mrc in enumerate(self.neg_mrc):
            i = i + self.n_pos
            self.indices.extend([(i, p) for p in range(mrc.data.shape[0])])
            self.mrcs[i] = mrc
            self.label[i] = 0.
        # Get length of set
        self.n = int(len(self.indices)*self.split)

        # Shuffle indices
        if self.shuffle:
            np.random.seed(1234)
            np.random.shuffle(self.indices)

        # Build train and validation generators
        self.train = _Generator(self.indices[:self.n], self)
        self.val = _Generator(self.indices[self.n:], self)

    def _get_indices(self, i):
        return self.indices[i * self.batch_size:i * self.batch_size + self.batch_size]

    @staticmethod
    def normalise(batch):
        mean = np.mean(batch, axis=(1, 2))
        std = np.std(batch, axis=(1, 2))
        batch = np.divide(np.subtract(batch.T, mean), std).T
        return batch

    def __getitem__(self, i):
        batch_img = []
        batch_rob = []
        batch_label = []
        for key, idx in self._get_indices(i):
            img = self.mrcs[key].data[idx]

            # Blur
            img = filters.gaussian(img, sigma=3) # sigma empirical - try it!

            # Resize
            img = transform.resize(img, (self.img_dim, self.img_dim))

            # Augment
            if self.data_augmentation:
                # Flip Left to Right
                if np.random.random() > 0.5:
                    img = np.fliplr(img)

                # Rotate 90° k times
                k = np.random.choice([0, 1, 2, 3], size=1)
                img = np.rot90(img, k=k)

            # Add image to batch
            batch_img.append(img)

            # Roberts filter
            img = filters.roberts(img)

            batch_rob.append(img)

            # Label
            batch_label.append(self.label[key])

        # Prepare as batch
        batch_img = self.normalise(np.stack(batch_img))
        batch_rob = self.normalise(np.stack(batch_rob))
        batch_label = np.stack(batch_label).reshape(self.batch_size, 1)

        # Convert to tensors
        # batch_img = tf.convert_to_tensor(batch_img)

        return batch_img, batch_rob, batch_label

    def close(self):
        for mrc in self.pos_mrc + self.neg_mrc:
            mrc.close()


if __name__ == "__main__":

    generator = DataGenerator(pos_files=["../data/pos.mrcs", "../data/pos_top.mrcs"],
                              neg_files=["../data/neg.mrcs"],
                              batch_size=12,
                             )

    train_X1, train_X2, train_y = generator.train[0]
    val_X1, val_X2, val_y = generator.val[0]

    print(train_X1.shape, train_X2.shape, train_y.shape)
    print(val_X1.shape, val_X2.shape, val_y.shape)

    generator.close()















