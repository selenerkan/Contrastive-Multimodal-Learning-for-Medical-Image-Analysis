from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import nibabel as nib

import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

from settings import IMAGE_PATH, FEATURES, TARGET, SEED
from torch.utils.data import DataLoader
import sys

import numpy as np
from monai import transforms
# import torchio as tio
import random


class Triplet_Loss_Dataset(Dataset):

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

    def preprocess(self, x):

        # preprocess img
        x = np.array(x, dtype=np.float32)

        # scale images between [0,1]
        min_val = x.min()
        x = (x - min_val) / (x.max() - min_val)
        x = torch.tensor(x)

        # create the channel dimension
        x = torch.unsqueeze(x, 0)
        x = self.transform(x)

        return x

    def __getitem__(self, idx):

        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        # get image name for the given index
        img_folder_name = self.tabular_data['image_id'][idx]

        # find a positive pair of the given image
        # get the label of the image
        label = self.tabular_data.loc[idx, self.target]
        # remove the current image form the tabular data
        tabular = self.tabular_data.drop(idx).reset_index()
        # get the positive and negative pair names
        positive_pairs = tabular[tabular[self.target]
                                 == label]['image_id'].unique()
        negative_pairs = tabular[tabular[self.target]
                                 != label]['image_id'].unique()
        # pick a random positive and negative image
        positive_img_folder_name = random.choice(positive_pairs)
        negative_img_folder_name = random.choice(negative_pairs)

        # get the index of the positive and negative pairs
        pos_idx = tabular.index[tabular['image_id']
                                == positive_img_folder_name].tolist()
        neg_idx = tabular.index[tabular['image_id']
                                == negative_img_folder_name].tolist()

        # get the paths for image and its positive and negative pairs
        img_path = os.path.join(
            self.imge_base_dir, img_folder_name + '.nii.gz')
        pos_img_path = os.path.join(
            self.imge_base_dir, positive_img_folder_name + '.nii.gz')
        neg_img_path = os.path.join(
            self.imge_base_dir, negative_img_folder_name + '.nii.gz')

        # load all three images
        image = nib.load(img_path)
        image = image.get_fdata()
        positive_img = nib.load(pos_img_path)
        positive_img = positive_img.get_fdata()
        negative_image = nib.load(neg_img_path)
        negative_image = negative_image.get_fdata()

        # Apply transformations
        transformed_images = self.preprocess(image)
        transformed_positive_images = self.preprocess(positive_img)
        transformed_negative_images = self.preprocess(negative_image)

        # get the tabular data for given index
        tab = self.X.iloc[idx].values
        positive_tab = self.X.iloc[pos_idx].values
        negative_tab = self.X.iloc[neg_idx].values

        return transformed_images, transformed_positive_images, transformed_negative_images, tab, positive_tab, negative_tab


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

    def __init__(self, csv_dir, loss_name='contrastive', n_views=2, age=None, batch_size=1, spatial_size=(120, 120, 120)):

        super().__init__()
        self.age = age
        self.csv_dir = csv_dir
        self.n_views = n_views
        self.batch_size = batch_size
        self.spatial_size = spatial_size
        self.loss_name = loss_name

        self.num_workers = 0
        if torch.cuda.is_available():
            self.num_workers = 16
        print(self.num_workers)

    def get_transforms(self):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        data_transforms = transforms.Compose([
            # tio.RandomElasticDeformation(p=0.5, num_control_points=(10),  # or just 7
            #                              locked_borders=0),
            # tio.RandomBiasField(p=0.5, coefficients=0.5, order=3),
            # tio.RandomSwap(p=0.6, patch_size=15, num_iterations=80),
            # tio.RandomGamma(p=0.5, log_gamma=(-0.3, 0.3))

            # MONAI TRANSFORMS
            # transforms.RandSpatialCrop(
            #     (80, 80, 80), random_center=True, random_size=False),
            # final image shape 160,120,120
            transforms.Resize(spatial_size=self.spatial_size),
            transforms.RandFlip(
                prob=0.5, spatial_axis=0),
            transforms.RandAdjustContrast(  # randomly change the contrast
                prob=0.5, gamma=(1.5, 2)),
            transforms.RandGaussianSmooth(
                sigma_x=(0.25, 1.5), prob=0.5),
            transforms.ToTensor(
                dtype=None, device=None, wrap_sequence=True, track_meta=None)
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
                                                                                                            test_size=0.2, random_state=SEED)

            self.subjects_train, self.subjects_val, self.labels_train, self.labels_val = train_test_split(self.subjects_train, self.labels_train,
                                                                                                          stratify=self.labels_train,
                                                                                                          test_size=0.25, random_state=SEED)
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

        if self.loss_name == 'contrastive':
            # create the dataset object using the dataframes created above for contrastive loss
            self.train = Contrastive_Dataset(self.train_df, image_base_dir=IMAGE_PATH,
                                             target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms(), self.n_views))

            self.test = Contrastive_Dataset(self.test_df, image_base_dir=IMAGE_PATH,
                                            target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms(), self.n_views))

            self.val = Contrastive_Dataset(self.val_df, image_base_dir=IMAGE_PATH,
                                           target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms(), self.n_views))

        else:
            # create the dataset object using the dataframes created above for triplet loss
            self.train = Triplet_Loss_Dataset(self.train_df, image_base_dir=IMAGE_PATH,
                                              target=TARGET, features=FEATURES, transform=self.get_transforms())

            self.test = Triplet_Loss_Dataset(self.test_df, image_base_dir=IMAGE_PATH,
                                             target=TARGET, features=FEATURES, transform=self.get_transforms())

            self.val = Triplet_Loss_Dataset(self.val_df, image_base_dir=IMAGE_PATH,
                                            target=TARGET, features=FEATURES, transform=self.get_transforms())

        return self.train_df, self.train

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):

        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


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
        min_val = x.min()
        x = (x - min_val) / (x.max() - min_val)

        x = torch.tensor(x)

        # create the channel dimension
        x = torch.unsqueeze(x, 0)

        return torch.stack([self.base_transform(x) for _ in range(self.n_views)])
