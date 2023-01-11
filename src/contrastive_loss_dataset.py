from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import nibabel as nib

import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

from settings import IMAGE_PATH, FEATURES, TARGET
from torch.utils.data import DataLoader
import sys

import numpy as np
from monai import transforms
# import torchio as tio


class Contrastive_Dataset(Dataset):

    def __init__(self, tabular_data, image_base_dir, target, features, transform=None):
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

    def __len__(self):

        return len(self.tabular)

    def __getitem__(self, idx):

        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        # get the tabular data for given index
        tab = self.X.iloc[idx].values

        # get image name for the given index
        img_folder_name = self.tabular_data['image_id'][idx]

        img_path = os.path.join(
            self.imge_base_dir, img_folder_name + '.nii.gz')

        image = nib.load(img_path)
        image = image.get_fdata()

        # Apply transformations
        transformed_images = self.transform(image)

        return transformed_images, tab


class ContrastiveDataModule(pl.LightningDataModule):

    def __init__(self, csv_dir, n_views=2, age=None, batch_size=1, spatial_size=(120, 120, 120)):

        super().__init__()
        self.age = age
        self.csv_dir = csv_dir
        self.n_views = n_views
        self.batch_size = batch_size
        self.spatial_size = spatial_size

    def get_transforms(self):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        data_transforms = transforms.Compose([
            # tio.RandomElasticDeformation(p=0.5, num_control_points=(10),  # or just 7
            #                              locked_borders=0),
            # tio.RandomBiasField(p=0.5, coefficients=0.5, order=3),
            # tio.RandomSwap(p=0.6, patch_size=15, num_iterations=80),
            # tio.RandomGamma(p=0.5, log_gamma=(-0.3, 0.3))

        ])

        return data_transforms

    def prepare_data(self):

        # read .csv to load the data
        self.tabular_data = pd.read_csv(self.csv_dir)

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

        # prepare test dataframe
        self.test_df = self.tabular_data[self.tabular_data['subject'].isin(
            self.subjects_test)].reset_index()

        # prepare val dataframe
        self.val_df = self.tabular_data[self.tabular_data['subject'].isin(
            self.subjects_val)].reset_index()

        # ----------------------------------------

        # create the dataset object using the dataframes created above
        self.train = Contrastive_Dataset(self.train_df, image_base_dir=IMAGE_PATH,
                                         target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms(), self.n_views))

        self.test = Contrastive_Dataset(self.test_df, image_base_dir=IMAGE_PATH,
                                        target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms(), self.n_views))

        self.val = Contrastive_Dataset(self.val_df, image_base_dir=IMAGE_PATH,
                                       target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms(), self.n_views))

        return self.train_df, self.train

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=16)

    def test_dataloader(self):

        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=16)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key.

    Params:
        - base_transform: the transform to apply
        - n_views: how many transforms of the same image to create

    Returns:
        - the stacked tensor of augmented images (shape: n_views x 1 x width x height x depth)
    """

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):

        # change the dtype
        x = np.array(x, dtype=np.float32)

        # scale images between [0,1]
        x = x / x.max()

        x = torch.tensor(x)

        # create the channel dimension
        x = torch.unsqueeze(x, 0)

        return torch.stack([self.base_transform(x) for _ in range(self.n_views)])
