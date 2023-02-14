from torch.utils.data import Dataset
import os
import pandas as pd
import nibabel as nib
import numpy as np

import pytorch_lightning as pl

from settings import IMAGE_PATH, TARGET, SEED, FEATURES
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys
from monai import transforms
from sklearn.model_selection import StratifiedKFold

import torch
import random


class Resnet_Dataset(Dataset):

    def __init__(self, tabular_data, image_base_dir, target, transform=None):
        """ initializes the dataset class for the resnet model

        tabular_data: The dataframe object holding the info about patients, the name of the MRI images and the labels

        image_base_dir:The directory of the folders containing the images

        target: name of the target feature in the tabular_data dataframe 

        transform:The trasformations for the input images

        """

        # TABULAR DATA
        # initialize the tabular data
        self.tabular_data = tabular_data.copy()

        # Save target
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


class Supervised_Multimodal_Dataset(Dataset):

    def __init__(self, tabular_data, image_base_dir, target, features, transform=None):
        """ initializes the dataset class for the supervised multimodal model

        tabular_data: The dataframe object holding the info about patients, the name of the MRI images and the labels

        image_base_dir:The directory of the folders containing the images

        target: name of the target feature in the tabular_data dataframe 

        features: subset feaures in the tabular data to be used in the model

        transform:The trasformations for the input images

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

        return len(self.tabular)

    def __getitem__(self, idx):

        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        label = self.y[idx]

        tab = self.X.iloc[idx].values

        # get image name in the given index
        img_folder_name = self.tabular_data['image_id'][idx]
        img_path = os.path.join(
            self.imge_base_dir, img_folder_name + '.nii.gz')

        image = nib.load(img_path)
        image = image.get_fdata()

        image = self.image_preprocess(image, self.transform)

        return image, tab, label


class Contrastive_Loss_Dataset(Dataset):

    def __init__(self, tabular_data, image_base_dir, target, features, transform=None):
        """ initializes the dataset class for the contrastive learning model using contrastive loss

        tabular_data: The dataframe object holding the info about patients, the name of the MRI images and the labels

        image_base_dir:The directory of the folders containing the images

        target: name of the target feature in the tabular_data dataframe 

        features: subset feaures in the tabular data to be used in the model

        transform:The trasformations for the input images

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


class Triplet_Loss_Dataset(Dataset):

    def __init__(self, tabular_data, image_base_dir, target, features, transform=None):
        """ initializes the dataset class for the contrastive learning model using triplet loss

        tabular_data: The dataframe object holding the info about patients, the name of the MRI images and the labels

        image_base_dir:The directory of the folders containing the images

        target: name of the target feature in the tabular_data dataframe 

        features: subset feaures in the tabular data to be used in the model

        transform:The trasformations for the input images

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

        return transformed_images, transformed_positive_images, transformed_negative_images, tab, positive_tab.squeeze(), negative_tab.squeeze(), label


class AdniDataModule(pl.LightningDataModule):

    def __init__(self, csv_dir, age=None, batch_size=1, spatial_size=(120, 120, 120)):

        super().__init__()
        self.age = age
        self.csv_dir = csv_dir
        self.batch_size = batch_size
        self.spatial_size = spatial_size

        self.num_workers = 8
        if torch.cuda.is_available():
            self.num_workers = 16
        print(self.num_workers)

    def get_transforms(self, spatial_size=(120, 120, 120)):
        """Return a set of data augmentation transformations"""
        data_transforms = transforms.Compose([
            # TORCHIO TRANSFORMATIONS
            # -------------------------------------------------------------------------------
            # tio.RandomElasticDeformation(p=0.5, num_control_points=(10), 
            #                              locked_borders=0),
            # tio.RandomBiasField(p=0.5, coefficients=0.5, order=3),
            # tio.RandomSwap(p=0.6, patch_size=15, num_iterations=80),
            # tio.RandomGamma(p=0.5, log_gamma=(-0.3, 0.3))

            # MONAI TRANSFORMS
            # ------------------------------------------------------------------------------
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

        # ONLY FOR OVERFITTING ON ONE IMAGE
        # self.train_df = self.train_df.iloc[:1]
        # self.val_df = self.train_df
        # self.test_df = self.train_df

        # print the patients in train and val
        print('number of patients in train: ', len(self.train_df))
        print('patient IDs in train: ', self.train_df.subject.unique())
        print('number of patients in val: ', len(self.val_df))
        print('patient IDs in val: ', self.val_df.subject.unique())

        # ----------------------------------------

    def set_resnet_dataset(self):
        # create the dataset object using the dataframes created above
        self.train = Resnet_Dataset(self.train_df, image_base_dir=IMAGE_PATH,
                                    target=TARGET, transform=transforms.Resize(spatial_size=self.spatial_size))

        self.test = Resnet_Dataset(self.test_df, image_base_dir=IMAGE_PATH,
                                   target=TARGET, transform=transforms.Resize(spatial_size=self.spatial_size))

        self.val = Resnet_Dataset(self.val_df, image_base_dir=IMAGE_PATH,
                                  target=TARGET, transform=transforms.Resize(spatial_size=self.spatial_size))

    def set_supervised_multimodal_dataloader(self):

        # create the dataset object using the dataframes created above
        self.train = Supervised_Multimodal_Dataset(self.train_df, image_base_dir=IMAGE_PATH,
                                                   target=TARGET, features=FEATURES, transform=transforms.Resize(spatial_size=self.spatial_size))

        self.test = Supervised_Multimodal_Dataset(self.test_df, image_base_dir=IMAGE_PATH,
                                                  target=TARGET, features=FEATURES, transform=transforms.Resize(spatial_size=self.spatial_size))

        self.val = Supervised_Multimodal_Dataset(self.val_df, image_base_dir=IMAGE_PATH,
                                                 target=TARGET, features=FEATURES, transform=transforms.Resize(spatial_size=self.spatial_size))

    def set_contrastive_loss_dataloader(self, n_views=2):
        self.train = Contrastive_Loss_Dataset(self.train_df, image_base_dir=IMAGE_PATH,
                                              target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms(), n_views))

        self.test = Contrastive_Loss_Dataset(self.test_df, image_base_dir=IMAGE_PATH,
                                             target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms(), n_views))

        self.val = Contrastive_Loss_Dataset(self.val_df, image_base_dir=IMAGE_PATH,
                                            target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms(), n_views))

    def set_triplet_loss_dataloader(self):
        self.train = Triplet_Loss_Dataset(self.train_df, image_base_dir=IMAGE_PATH,
                                          target=TARGET, features=FEATURES, transform=self.get_transforms())

        self.test = Triplet_Loss_Dataset(self.test_df, image_base_dir=IMAGE_PATH,
                                         target=TARGET, features=FEATURES, transform=self.get_transforms())

        self.val = Triplet_Loss_Dataset(self.val_df, image_base_dir=IMAGE_PATH,
                                        target=TARGET, features=FEATURES, transform=self.get_transforms())

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):

        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# WILL BE CHECKED LATER
class KfoldMultimodalDataModule(pl.LightningDataModule):

    def __init__(self, csv_dir, fold_number=2, age=None, batch_size=1):

        super().__init__()
        self.age = age
        self.csv_dir = csv_dir
        self.fold_number = fold_number
        self.batch_size = batch_size

    def prepare_data(self):

        # read .csv to load the data
        self.tabular_data = pd.read_csv(self.csv_dir)

        # filter the dataset with the given age
        if self.age is not None:
            self.tabular_data = self.tabular_data[self.tabular_data.age == self.age]
            self.tabular_data = self.tabular_data.reset_index()

        # split data into kfolds
        skf = StratifiedKFold(n_splits=self.fold_number,
                              random_state=None, shuffle=False)

        # print(self.tabular_data.label_numeric)
        train_dataloaders = []
        val_dataloaders = []
        for i, (train_index, val_index) in enumerate(skf.split(self.tabular_data, self.tabular_data.label_numeric)):
            # filter the data by the generated index
            self.train_df = self.tabular_data.filter(
                items=train_index, axis=0).reset_index()
            self.val_df = self.tabular_data.filter(
                items=val_index, axis=0).reset_index()

            # create datasets from these dataframes
            self.train = Supervised_Multimodal_Dataset(self.train_df, image_base_dir=IMAGE_PATH,
                                                       target=TARGET, features=FEATURES)

            self.val = Supervised_Multimodal_Dataset(self.val_df, image_base_dir=IMAGE_PATH,
                                                     target=TARGET, features=FEATURES)

            # create dataloaders and add them to a list
            train_dataloaders.append(DataLoader(
                self.train, batch_size=self.batch_size, shuffle=True, num_workers=16))
            val_dataloaders.append(DataLoader(
                self.val, batch_size=self.batch_size, shuffle=True, num_workers=16))

        return train_dataloaders, val_dataloaders


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
