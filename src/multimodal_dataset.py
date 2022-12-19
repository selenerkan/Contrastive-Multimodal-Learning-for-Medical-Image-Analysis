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

    def __init__(self, tabular_data, image_base_dir, target, features, categorical=None, transform=None, target_transform=None, age=None):
        """

        csv_dir: The directiry for the .csv file (tabular data) including the labels

        image_base_dir:The directory of the folders containing the images

        transform:The trasformations for the input images

        Target_transform:The trasformations for the target(label)

        """
        # TABULAR DATA
        # initialize the tabular data
        self.tabular_data = tabular_data.copy()

        # keep relevant features in the tabular data
        self.features = features
        self.tabular = self.tabular_data[self.features]

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

    def __init__(self, csv_dir, age=None):

        super().__init__()
        self.age = age
        self.csv_dir = csv_dir

    def prepare_data(self):

        # read .csv to load the data
        self.tabular_data = pd.read_csv(self.csv_dir)

        # filter the dataset with the given age
        if self.age is not None:
            self.tabular_data = self.tabular[self.tabular_data.age == self.age]
            self.tabular_data = self.tabular_data.reset_index()

        # ----------------------------------------
        # split the data by patient ID

        # get unique patient and label pairs
        patient_label_list = self.tabular_data.groupby(
            'subject')['label_numeric'].unique()
        patient_label_df = pd.DataFrame(patient_label_list)
        patient_label_df = patient_label_df.reset_index()

        # make stratified split on the labels
        # get the subjects and labels fir train, test, validation
        self.subjects_train, self.subjects_test, self.labels_train, self.labels_test = train_test_split(patient_label_df.subject, patient_label_df.label_numeric,
                                                                                                        stratify=patient_label_df.label_numeric,
                                                                                                        test_size=0.2)

        self.subjects_train, self.subjects_val, self.labels_train, self.labels_val = train_test_split(self.subjects_train, self.labels_train,
                                                                                                      stratify=self.labels_train,
                                                                                                      test_size=0.25)
        # ----------------------------------------
        # prepare the train, test, validation datasets using the subjects assigned to them

        # prepare train dataframe
        self.train_df = self.tabular_data[self.tabular_data['subject'].isin(
            self.subjects_train)]

        # prepare test dataframe
        self.test_df = self.tabular_data[self.tabular_data['subject'].isin(
            self.subjects_test)]

        # prepare val dataframe
        self.val_df = self.tabular_data[self.tabular_data['subject'].isin(
            self.subjects_val)]

        # ----------------------------------------

        # create the dataset object using the dataframes created above
        self.train = Multimodal_Dataset(self.train_df, image_base_dir=IMAGE_PATH,
                                        target=TARGET, features=FEATURES,
                                        transform=transformation, target_transform=target_transformations, age=self.age)

        self.test = Multimodal_Dataset(self.test_df, image_base_dir=IMAGE_PATH,
                                       target=TARGET, features=FEATURES,
                                       transform=transformation, target_transform=target_transformations, age=self.age)

        self.val = Multimodal_Dataset(self.val_df, image_base_dir=IMAGE_PATH,
                                      target=TARGET, features=FEATURES,
                                      transform=transformation, target_transform=target_transformations, age=self.age)

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=1, shuffle=True)
        # return DataLoader(self.X_train, batch_size=1, shuffle=True)

    def val_dataloader(self):

        return DataLoader(self.val, batch_size=1, shuffle=False)
        # return DataLoader(self.X_valid, batch_size=1, shuffle=False)

    def test_dataloader(self):

        return DataLoader(self.test, batch_size=1, shuffle=False)
        # return DataLoader(self.X_test, batch_size=1, shuffle=False)
