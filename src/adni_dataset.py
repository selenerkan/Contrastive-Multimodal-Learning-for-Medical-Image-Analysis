from torch.utils.data import Dataset
import os
import pandas as pd
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
from adni_settings import image_dir, root_dir, test_dir, train_dir, TARGET, SEED, FEATURES
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from monai import transforms

import torch
import random


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
        img_folder_name = self.tabular_data['image_original'][idx]
        img_path = os.path.join(
            self.imge_base_dir, img_folder_name)

        image = nib.load(img_path)
        image = image.get_fdata()

        image = self.image_preprocess(image, self.transform)

        return image, tab, label


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
        self.features = features.copy()
        self.features.remove(target)
        # self.tabular = self.tabular_data[self.features]

        # Save target and predictors
        self.target = target
        # self.X = self.tabular.drop(self.target, axis=1)
        # self.y = self.tabular[self.target]

        # IMAGE DATA
        self.imge_base_dir = image_base_dir
        self.transform = transform

    def __len__(self):

        return len(self.tabular_data)

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

    def load_pairs(self, idx):
        # remove the data belongs to the sample individual
        label_anchor = self.tabular_data[self.target].iloc[idx]
        p_id = self.tabular_data['p_id'].iloc[idx]
        tabular = self.tabular_data[self.tabular_data['p_id'] != p_id]

        # get the positive and negative pair names
        # dont get the images form the same person as positive pairs?
        positive_pairs = tabular[tabular.label_numeric
                                 == label_anchor].drop_duplicates()
        negative_pairs = tabular[tabular.label_numeric
                                 != label_anchor].drop_duplicates()

        # pick a random positive and negative image
        pos_idx = int(torch.randint(len(positive_pairs), (1,)))
        neg_idx = int(torch.randint(len(negative_pairs), (1,)))

        positive_img_name = positive_pairs['image_original'].iloc[pos_idx]
        pos_img_path = os.path.join(
            self.imge_base_dir, positive_img_name)

        negative_img_name = negative_pairs['image_original'].iloc[neg_idx]
        neg_img_path = os.path.join(
            self.imge_base_dir, negative_img_name)

        positive_image = nib.load(pos_img_path)
        positive_image = positive_image.get_fdata()
        negative_image = nib.load(neg_img_path)
        negative_image = negative_image.get_fdata()

        # get the tabular data for given index
        positive_tab = positive_pairs[self.features].iloc[pos_idx].values
        negative_tab = negative_pairs[self.features].iloc[neg_idx].values

        return positive_image, positive_tab, negative_image, negative_tab

    def __getitem__(self, idx):

        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        # get the label and tabular features of the sample
        label_anchor = self.tabular_data[self.target].iloc[idx]
        tab_anchor = self.tabular_data[self.features].iloc[idx].values

        # get image name for the given index
        img_name = self.tabular_data['image_original'][idx]
        img_path = os.path.join(self.imge_base_dir, img_name)

        # load all three images
        image = nib.load(img_path)
        image = image.get_fdata()

        positive_image, positive_tab, negative_image, negative_tab = self.load_pairs(
            idx)

        if self.transform:
            transformed_images = self.preprocess(image)
            transformed_positive_images = self.preprocess(positive_image)
            transformed_negative_images = self.preprocess(negative_image)

        return transformed_images, transformed_positive_images, transformed_negative_images, tab_anchor, positive_tab, negative_tab, label_anchor


class AdniDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=1, spatial_size=(64, 64, 64)):

        super().__init__()
        self.batch_size = batch_size
        self.spatial_size = spatial_size

        self.num_workers = 0
        if torch.cuda.is_available():
            self.num_workers = 16
        print('number of workers: ', self.num_workers)

    def get_transforms(self):
        """Return a set of data augmentation transformations"""
        transform_train = transforms.Compose([
            # TORCHIO TRANSFORMATIONS
            # -------------------------------------------------------------------------------
            # tio.RandomElasticDeformation(p=0.5, num_control_points=(10),
            #                              locked_borders=0),
            # tio.RandomBiasField(p=0.5, coefficients=0.5, order=3),
            # tio.RandomSwap(p=0.6, patch_size=15, num_iterations=80),
            # tio.RandomGamma(p=0.5, log_gamma=(-0.3, 0.3))

            # MONAI TRANSFORMS
            # ------------------------------------------------------------------------------
            # transforms.Resize(spatial_size=self.spatial_size),
            transforms.RandFlip(
                prob=0.5, spatial_axis=0),
            transforms.RandAdjustContrast(  # randomly change the contrast
                prob=0.5, gamma=(1.5, 2)),
            transforms.RandGaussianSmooth(
                sigma_x=(0.25, 1.5), prob=0.5),
            transforms.ToTensor(
                dtype=None, device=None, wrap_sequence=True, track_meta=None)
        ])
        transform_val = transforms.Compose([
            # transforms.Resize(spatial_size=self.spatial_size),
            transforms.ToTensor(
                dtype=None, device=None, wrap_sequence=True, track_meta=None)
        ])
        return {'train': transform_train, 'val': transform_val}

    def prepare_data(self, seed):

        # read .csv to load the data
        self.train_data = pd.read_csv(root_dir + train_dir)
        self.test_df = pd.read_csv(root_dir + test_dir)

        # # filter the dataset with the given age
        # if self.age is not None:
        #     self.tabular_data = self.tabular_data[self.tabular_data.age == self.age]
        #     self.tabular_data = self.tabular_data.reset_index()

        # split train data into train and val
        # ----------------------------------------
        # split the data by patient ID
        train_patient_label_list = self.train_data.groupby(
            'p_id')['label_numeric'].first()
        train_patient_label_df = pd.DataFrame(train_patient_label_list)
        train_patient_label_df = train_patient_label_df.reset_index()

        # get stritified split for train and val
        ss = StratifiedSampler(torch.FloatTensor(
            train_patient_label_df.label_numeric), test_size=0.2, seed=seed)
        train_indices, val_indices = ss.gen_sample_array()

        # store indices of train, test, valin a dictionary
        indices = {'train': train_indices,
                   'val': val_indices
                   }

        # get the patient ids using the generated indices
        self.train_patients = train_patient_label_df.iloc[indices['train']]
        self.val_patients = train_patient_label_df.iloc[indices['val']]

        # prepare the train, test, validation datasets using the subjects assigned to them
        # prepare train dataframe
        self.train_df = self.train_data[self.train_data['p_id'].isin(
            self.train_patients.p_id)].reset_index(drop=True)
        # prepare val dataframe
        self.val_df = self.train_data[self.train_data['p_id'].isin(
            self.val_patients.p_id)].reset_index(drop=True)

        # # ONLY FOR OVERFITTING ON ONE IMAGE
        # self.train_df = self.train_df.groupby(
        #     'label').apply(lambda x: x.sample(5)).droplevel(0).reset_index(drop=True)
        # self.val_df = self.val_df.groupby(
        #     'label').apply(lambda x: x.sample(5)).droplevel(0).reset_index(drop=True)
        # self.test_df = self.test_df.groupby(
        #     'label').apply(lambda x: x.sample(5)).droplevel(0).reset_index(drop=True)

        print('number of patients in train: ', len(self.train_df))
        print('patient IDs in train: ', self.train_df.p_id.unique())
        print('number of patients in val: ', len(self.val_df))
        print('patient IDs in val: ', self.val_df.p_id.unique())

    def prepare_zero_shot_data(self, seed, percent):

        # read .csv to load the data
        self.train_data = pd.read_csv(root_dir + train_dir)
        self.test_df = pd.read_csv(root_dir + test_dir)
        self.percent = percent

        # ----------------------------------------
        # split the data by patient ID
        train_patient_label_list = self.train_data.groupby(
            'p_id')['label_numeric'].first()
        train_patient_label_df = pd.DataFrame(train_patient_label_list)
        train_patient_label_df = train_patient_label_df.reset_index()

        # get stritified split for train and val
        ss = StratifiedSampler(torch.FloatTensor(
            train_patient_label_df.label_numeric), test_size=self.percent, seed=seed)
        train_indices, val_indices = ss.gen_sample_array()

        # store indices of train, test, valin a dictionary
        indices = {'train': val_indices,
                   'val': train_indices
                   }
        # only_train_df = pd.DataFrame(
        #     train_patient_label_df.iloc[train_indices])
        # print(only_train_df)
        # print(len(only_train_df))
        # print(only_train_df.label_numeric)

        # split the train set again to keep only x% data
        # ss2 = StratifiedSampler(torch.FloatTensor(
        #     only_train_df.label_numeric), test_size=self.percent, seed=seed)
        # train_remaining_indices, train_final_indices = ss2.gen_sample_array()

        # # store indices of train, test, valin a dictionary
        # indices = {'train': train_final_indices,
        #            'val': val_indices
        #            }

        # get the patient ids using the generated indices
        self.train_patients = train_patient_label_df.iloc[indices['train']]
        self.val_patients = train_patient_label_df.iloc[indices['val']]

        # prepare the train, test, validation datasets using the subjects assigned to them
        # prepare train dataframe
        self.train_df = self.train_data[self.train_data['p_id'].isin(
            self.train_patients.p_id)].reset_index(drop=True)
        # prepare val dataframe
        self.val_df = self.train_data[self.train_data['p_id'].isin(
            self.val_patients.p_id)].reset_index(drop=True)

        print('number of patients in train: ', len(self.train_df))
        print('patient IDs in train: ', self.train_df.p_id.unique())
        print('# samples in each class (TRAIN): ',
              self.train_df.groupby('label_numeric').count())
        print('number of patients in val: ', len(self.val_df))
        print('patient IDs in val: ', self.val_df.p_id.unique())

        # ----------------------------------------
    def set_supervised_multimodal_dataloader(self):

        # create the dataset object using the dataframes created above
        self.train = Supervised_Multimodal_Dataset(self.train_df, image_base_dir=image_dir,
                                                   target=TARGET, features=FEATURES, transform=self.get_transforms()['train'])

        self.test = Supervised_Multimodal_Dataset(self.test_df, image_base_dir=image_dir,
                                                  target=TARGET, features=FEATURES, transform=self.get_transforms()['val'])

        self.val = Supervised_Multimodal_Dataset(self.val_df, image_base_dir=image_dir,
                                                 target=TARGET, features=FEATURES, transform=self.get_transforms()['val'])

    def set_contrastive_loss_dataloader(self, n_views=2):
        self.train = Supervised_Multimodal_Dataset(self.train_df, image_base_dir=image_dir,
                                                   target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms()['train'], n_views))

        self.test = Supervised_Multimodal_Dataset(self.test_df, image_base_dir=image_dir,
                                                  target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms()['val'], n_views))

        self.val = Supervised_Multimodal_Dataset(self.val_df, image_base_dir=image_dir,
                                                 target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms()['val'], n_views))

    def set_triplet_loss_dataloader(self):
        self.train = Triplet_Loss_Dataset(self.train_df, image_base_dir=image_dir,
                                          target=TARGET, features=FEATURES, transform=self.get_transforms()['train'])

        self.test = Triplet_Loss_Dataset(self.test_df, image_base_dir=image_dir,
                                         target=TARGET, features=FEATURES, transform=self.get_transforms()['val'])

        self.val = Triplet_Loss_Dataset(self.val_df, image_base_dir=image_dir,
                                        target=TARGET, features=FEATURES, transform=self.get_transforms()['val'])

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
        return torch.stack([self.base_transform(x) for _ in range(self.n_views)])


class Sampler(object):
    """Base class for all Samplers.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes
    """

    def __init__(self, class_vector, test_size, seed, n_splits=1):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = n_splits
        self.class_vector = class_vector
        self.test_size = test_size
        self.seed = seed

    def gen_sample_array(self):

        s = StratifiedShuffleSplit(
            n_splits=self.n_splits, test_size=self.test_size, random_state=self.seed)
        X = torch.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return train_index, test_index

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)
