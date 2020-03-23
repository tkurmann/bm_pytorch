from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
import logging
import random
logger = logging.getLogger(__name__)


class OCTHDF5Dataset(Dataset):
    """Instrument dataset."""

    def __init__(self, hdf5_file, image_set, label_set, transform_image=None):


        self.dataset = None
        self.hdf5_file = hdf5_file
        self.image_set_name = image_set
        self.label_set_name = label_set

        self.image_set = None
        self.label_set = None
        with h5py.File(self.hdf5_file, 'r') as file:
            self.dataset_len = file[image_set].shape[0]

        self.transform_image = transform_image

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        if self.image_set is None:
            self.image_set = h5py.File(self.hdf5_file, 'r')[self.image_set_name]

        if self.label_set is None:
            self.label_set = h5py.File(self.hdf5_file, 'r')[self.label_set_name]


        if torch.is_tensor(idx):
            idx = idx.tolist()


        image = self.image_set[idx]
        label = self.label_set[idx]
        image = np.expand_dims(image, axis=-1)
        image = np.concatenate([image, image, image], axis=-1)


        seed = random.randint(0,2**32)
        if self.transform_image:
            random.seed(seed)
            image = self.transform_image(image)


        sample = {'images': image,  'label': label}
        return sample
