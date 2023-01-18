from torch.utils.data import Dataset
import os
import pandas as pd
import nibabel as nib
import numpy as np

import pytorch_lightning as pl

from settings import IMAGE_PATH, TARGET
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys
from monai import transforms

import torch


class Adni_Dataset(Dataset):

    def __init__(self, tabular_data, image_base_dir, target, transform=None):
        """

        csv_dir: The directiry for the .csv file holding the name of the images and the labels

        image_base_dir:The directory of the folders containing the images

        transform:The trasformations for the input images

        Target_transform:The trasformations for the target(label)

        """

        # TABULAR DATA
        # initialize the tabular data
        self.tabular_data = tabular_data.copy()

        # Save target and predictors
        self.target = target
        self.y = self.tabular_data[self.target]

        # IMAGE DATA
        self.imge_base_dir = image_base_dir
        self.transform = transform

    def image_preprocess(self, image, transform):

        # change to numpy and scale images between [0,1]
        image = np.array(image, dtype=np.float32)
        min_val = image.min()
        image = (image - min_val) / (image.max() - min_val)

        image = torch.tensor(image)

        # create the channel dimension
        image = torch.unsqueeze(image, 0)

        if transform:
            image = transform(image)

        return image

    def __len__(self):

        return len(self.tabular_data)

    def __getitem__(self, idx):

        # get the label
        label = self.y[idx]

        # get image name in the given index
        img_folder_name = self.tabular_data['image_id'][idx]
        img_path = os.path.join(
            self.imge_base_dir, img_folder_name + '.nii.gz')

        image = nib.load(img_path)
        image = image.get_fdata()
        image = self.image_preprocess(image, self.transform)

        if self.transform:
            image = self.transform(image)

        return image, label


class AdniDataModule(pl.LightningDataModule):

    def __init__(self, csv_file, age=None, batch_size=1, spatial_size=(120, 120, 120)):

        super().__init__()
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.age = age
        self.spatial_size = spatial_size

        self.num_workers = 0
        if torch.cuda.is_available():
            self.num_workers = 16
        print(self.num_workers)

    def prepare_data(self):

        # read .csv to load the data
        self.tabular_data = pd.read_csv(self.csv_file)

        # filter the dataset with the given age
        if self.age is not None:
            self.tabular_data = self.tabular_data[self.tabular_data.age == self.age]
            self.tabular_data = self.tabular_data.reset_index()

        # ----------------------------------------
        # split the data by patient ID

        # get unique patient and label pairs
        patient_label_list = self.tabular_data.groupby(
            'subject')['label_numeric'].unique()
        patient_label_df = pd.DataFrame(patient_label_list)
        patient_label_df = patient_label_df.reset_index()

        try:
            # make stratified split on the labels
            # get the subjects and labels fir train, test, validation
            self.subjects_train, self.subjects_test, self.labels_train, self.labels_test = train_test_split(patient_label_df.subject, patient_label_df.label_numeric,
                                                                                                            stratify=patient_label_df.label_numeric,
                                                                                                            test_size=0.2)

            self.subjects_train, self.subjects_val, self.labels_train, self.labels_val = train_test_split(self.subjects_train, self.labels_train,
                                                                                                          stratify=self.labels_train,
                                                                                                          test_size=0.25)
        except Exception as e:
            print('Dataset couldn\'t be split by patient. Possible cause is having only 1 patient in test or validation')
            print(e)
            sys.exit(e)
        # ----------------------------------------
        # prepare the train, test, validation datasets using the subjects assigned to them

        # prepare train dataframe
        self.train_df = self.tabular_data[self.tabular_data['subject'].isin(
            self.subjects_train)].reset_index()

        # ONLY FOR OVERFITTING ON ONE IMAGE
        # self.train_df = self.train_df.iloc[:1]

        # print the patients in train
        print('number of patients in train: ', len(self.train_df))

        # prepare test dataframe
        self.test_df = self.tabular_data[self.tabular_data['subject'].isin(
            self.subjects_test)].reset_index()

        # ONLY FOR OVERFITTING ON ONE IMAGE
        # self.test_df = self.train_df

        # prepare val dataframe
        self.val_df = self.tabular_data[self.tabular_data['subject'].isin(
            self.subjects_val)].reset_index()

        # ONLY FOR OVERFITTING ON ONE IMAGE
        # self.val_df = self.train_df

        # print the patients in train
        print('number of patients in val: ', len(self.val_df))

        # ----------------------------------------

        # create the dataset object using the dataframes created above
        self.train = Adni_Dataset(self.train_df, image_base_dir=IMAGE_PATH,
                                  target=TARGET, transform=self.get_transforms(self.spatial_size))

        self.test = Adni_Dataset(self.test_df, image_base_dir=IMAGE_PATH,
                                 target=TARGET, transform=self.get_transforms(self.spatial_size))

        self.val = Adni_Dataset(self.val_df, image_base_dir=IMAGE_PATH,
                                target=TARGET, transform=self.get_transforms(self.spatial_size))

    def get_transforms(self, spatial_size=(120, 120, 120)):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        data_transforms = None
        if spatial_size:
            data_transforms = transforms.Compose([
                transforms.Resize(spatial_size=spatial_size)
            ])
        return data_transforms

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):

        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
