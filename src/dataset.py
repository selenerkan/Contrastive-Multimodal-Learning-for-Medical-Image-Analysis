from torch.utils.data import Dataset
import os
import pandas as pd
import nibabel as nib
import numpy as np

import pytorch_lightning as pl

import torch

from settings import CSV_FILE, IMAGE_PATH, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, transformation, target_transformations
from torch.utils.data import DataLoader


class Adni_Dataset(Dataset):

    def __init__(self, csv_dir, image_base_dir, transform=None, target_transform=None):
        """

        csv_dir: The directiry for the .csv file holding the name of the images and the labels

        image_base_dir:The directory of the folders containing the images

        transform:The trasformations for the input images

        Target_transform:The trasformations for the target(label)

        """

        # read .csv to get image naems and their labels
        self.img_list = pd.read_csv(csv_dir)

        self.imge_base_dir = image_base_dir

        self.transform = transform

        self.target_transform = target_transform

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, idx):

        img_path = os.path.join(
            self.imge_base_dir, self.img_list['image'][idx])

        image = nib.load(img_path)
        image = image.get_fdata()

        # change to numpy
        image = np.array(image, dtype=np.float32)

        # scale images between [0,1]
        image = image / image.max()

        # get the label
        label = self.img_list['label'][idx]

        if self.transform:

            image = self.transform(image)

        if self.target_transform:

            label = self.target_transform(label)

        return image, label


class AdniDataModule(pl.LightningDataModule):

    def __init__(self):

        super().__init__()

    def prepare_data(self):

        self.train = Adni_Dataset(CSV_FILE + r'\train.csv', IMAGE_PATH,
                                  transformation, target_transformations)

        self.valid = Adni_Dataset(CSV_FILE + r'\val.csv', IMAGE_PATH,
                                  transformation, target_transformations)

        self.test = Adni_Dataset(CSV_FILE + r'\test.csv', IMAGE_PATH,
                                 transformation, target_transformations)

        # self.train, self.valid = torch.utils.data.random_split(
        #     self.train, [TRAIN_SIZE, VAL_SIZE + TEST_SIZE])

        # self.valid, self.test = torch.utils.data.random_split(
        #     self.valid, [VAL_SIZE, TEST_SIZE])

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=1, shuffle=True)

    def val_dataloader(self):

        return DataLoader(self.valid, batch_size=1, shuffle=False)

    def test_dataloader(self):

        return DataLoader(self.test, batch_size=1, shuffle=False)
