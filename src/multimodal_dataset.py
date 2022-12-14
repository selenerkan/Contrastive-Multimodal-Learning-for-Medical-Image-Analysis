from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import nibabel as nib

import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

from settings import CSV_FILE, IMAGE_PATH, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, FEATURES, TARGET, transformation, target_transformations
from torch.utils.data import DataLoader


class Multimodal_Dataset(Dataset):

    def __init__(self, csv_dir, image_base_dir, target, features, categorical=None, transform=None, target_transform=None):
        """

        csv_dir: The directiry for the .csv file (tabular data) including the labels

        image_base_dir:The directory of the folders containing the images

        transform:The trasformations for the input images

        Target_transform:The trasformations for the target(label)

        """
        # TABULAR DATA
        # read .csv to load the data
        self.multimodal = pd.read_csv(csv_dir)

        # keep relevant features in the tabular data
        self.features = features
        self.tabular = self.multimodal[self.features]

        # Save target and predictors
        self.target = target
        self.X = self.tabular.drop(self.target, axis=1)
        self.y = self.tabular[self.target]

        # IMAGE DATA
        self.imge_base_dir = image_base_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):

        return len(self.tabular)

    def __getitem__(self, idx):

        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        label = self.y[idx]

        tab = self.X.iloc[idx].values

        # get image name in the given index
        img_folder_name = self.multimodal['image_id'][idx]

        img_path = os.path.join(
            self.imge_base_dir, img_folder_name + '.nii.gz')

        image = nib.load(img_path)
        image = image.get_fdata()

        # change to numpy
        image = np.array(image, dtype=np.float32)

        # scale images between [0,1]
        image = image / image.max()

        return image, tab, label


class MultimodalDataModule(pl.LightningDataModule):

    def __init__(self):

        super().__init__()

    def prepare_data(self):

        self.train = Multimodal_Dataset(csv_dir=CSV_FILE + r'\train.csv', image_base_dir=IMAGE_PATH,
                                        target=TARGET, features=FEATURES,
                                        transform=transformation, target_transform=target_transformations)

        self.valid = Multimodal_Dataset(csv_dir=CSV_FILE + r'\val.csv', image_base_dir=IMAGE_PATH,
                                        target=TARGET, features=FEATURES,
                                        transform=transformation, target_transform=target_transformations)

        self.test = Multimodal_Dataset(csv_dir=CSV_FILE + r'\test.csv', image_base_dir=IMAGE_PATH,
                                       target=TARGET, features=FEATURES,
                                       transform=transformation, target_transform=target_transformations)

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.train.X, self.train.y,
        #                                                                         stratify=self.train.y,
        #                                                                         test_size=0.2)

        # self.X_train, self.self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
        #                                                                            stratify=self.train.y,
        #                                                                            test_size=0.25)

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=1, shuffle=True)
        # return DataLoader(self.X_train, batch_size=1, shuffle=True)

    def val_dataloader(self):

        return DataLoader(self.valid, batch_size=1, shuffle=False)
        # return DataLoader(self.X_valid, batch_size=1, shuffle=False)

    def test_dataloader(self):

        return DataLoader(self.test, batch_size=1, shuffle=False)
        # return DataLoader(self.X_test, batch_size=1, shuffle=False)
