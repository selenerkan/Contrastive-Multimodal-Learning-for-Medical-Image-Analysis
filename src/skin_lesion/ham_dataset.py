from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import numpy as np

import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit
from ham_settings import image_dir, TARGET, FEATURES, root_dir, train_dir, test_dir
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys
import torchvision.transforms as transforms
import torch
from torchvision.io import read_image

import random


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
        self.features.remove('label')

        # Save target and predictors
        self.target = target

        # IMAGE DATA
        self.imge_base_dir = image_base_dir
        self.transform = transform

    def __len__(self):

        return len(self.tabular_data)

    def load_pairs(self, idx):
        # remove the data belongs to the sample individual
        label_anchor = self.tabular_data[self.target].iloc[idx]
        lesion_id = self.tabular_data['lesion_id'].iloc[idx]
        tabular = self.tabular_data[self.tabular_data['lesion_id'] != lesion_id]

        # get the positive and negative pair names
        # dont get the images form the same person as positive pairs?
        positive_pairs = tabular[tabular.label
                                 == label_anchor].drop_duplicates()
        negative_pairs = tabular[tabular.label
                                 != label_anchor].drop_duplicates()

        # pick a random positive and negative image
        pos_idx = int(torch.randint(len(positive_pairs), (1,)))
        neg_idx = int(torch.randint(len(negative_pairs), (1,)))

        positive_img_name = positive_pairs['image_id'].iloc[pos_idx]
        pos_img_folder_name = positive_pairs.dx.iloc[pos_idx]
        pos_img_path = os.path.join(
            self.imge_base_dir, pos_img_folder_name, positive_img_name + '.jpg')

        negative_img_name = negative_pairs['image_id'].iloc[neg_idx]
        neg_img_folder_name = negative_pairs.dx.iloc[neg_idx]
        neg_img_path = os.path.join(
            self.imge_base_dir, neg_img_folder_name, negative_img_name + '.jpg')

        positive_image = Image.open(pos_img_path)
        positive_image = positive_image.convert("RGB")
        negative_image = Image.open(neg_img_path)
        negative_image = negative_image.convert("RGB")

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
        img_folder_name = self.tabular_data.dx[idx]
        img_name = self.tabular_data['image_id'][idx]
        img_path = os.path.join(
            self.imge_base_dir, img_folder_name, img_name + '.jpg')

        # load all three images
        image = Image.open(img_path)
        image = image.convert("RGB")

        positive_image, positive_tab, negative_image, negative_tab = self.load_pairs(
            idx)

        if self.transform:
            transformed_images = self.transform(image)
            transformed_positive_images = self.transform(positive_image)
            transformed_negative_images = self.transform(negative_image)

        return transformed_images, transformed_positive_images, transformed_negative_images, tab_anchor, positive_tab, negative_tab, label_anchor


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

    def __len__(self):

        return len(self.tabular)

    def __getitem__(self, idx):

        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        label = self.y[idx]
        tab = self.X.iloc[idx].values

        # get image name in the given index
        img_folder_name = self.tabular_data.dx[idx]
        img_name = self.tabular_data['image_id'][idx]
        img_path = os.path.join(
            self.imge_base_dir, img_folder_name, img_name + '.jpg')
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, tab, label


class HAMDataModule(pl.LightningDataModule):

    def __init__(self, csv_dir, age=None, batch_size=1):

        super().__init__()
        self.age = age
        self.csv_dir = csv_dir
        self.batch_size = batch_size
        self.n_views = 2

        self.num_workers = 0
        if torch.cuda.is_available():
            self.num_workers = 16
        print(self.num_workers)

    def get_transforms(self):
        """Return a set of data augmentation transformations"""

        # normalization values for pretrained resnet on Imagenet
        norm_mean = (0.4914, 0.4822, 0.4465)
        norm_std = (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose([
            # transforms.PILToTensor(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=60),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        return {'train': transform_train, 'val': transform_val}

    # This code was also doing the test split
    # It is used once only to get the test data and store it in a .csv file
    # USE THE CODE AFTER THIS COMMENT BLOCK FOR PREPARE DATA FUNCTION
    # def prepare_data(self):

    #     # PREVIOUS
    #     # --------------------------------------------------
    #     # read .csv to load the data
    #     self.tabular_data = pd.read_csv(self.csv_dir)

    #     # filter the dataset with the given age
    #     if self.age is not None:
    #         self.tabular_data = self.tabular_data[self.tabular_data.age == self.age]
    #         self.tabular_data = self.tabular_data.reset_index(drop=True)

    #     # ----------------------------------------
    #     # split the data by patient ID

    #     # get unique patient and label pairs
    #     patient_label_list = self.tabular_data.groupby(
    #         'lesion_id')['label'].first()
    #     patient_label_df = pd.DataFrame(patient_label_list)
    #     patient_label_df = patient_label_df.reset_index()

    #     # get stritified split for train and test
    #     ss = StratifiedSampler(torch.FloatTensor(
    #         patient_label_df.label), test_size=0.2)
    #     pre_train_indices, test_indices = ss.gen_sample_array()

    #     # get the stritified split for train and val
    #     # train_label = np.delete(patient_label_df.label, test_indices, None)
    #     train_label = patient_label_df.label.drop(index=test_indices)
    #     ss = StratifiedSampler(torch.FloatTensor(train_label), test_size=0.2)
    #     train_indices, val_indices = ss.gen_sample_array()

    #     # store indices of train, test, valin a dictionary
    #     indices = {'train': pre_train_indices[train_indices],  # Indices of second sampler are used on pre_train_indices
    #                # Indices of second sampler are used on pre_train_indices
    #                'val': pre_train_indices[val_indices],
    #                'test': test_indices
    #                }

    #     # get the patient ids using the generated indices
    #     self.train_patients = patient_label_df.iloc[indices['train']]
    #     self.val_patients = patient_label_df.iloc[indices['val']]
    #     self.test_patients = patient_label_df.iloc[indices['test']]

    #     # ----------------------------------------

    #     # prepare the train, test, validation datasets using the subjects assigned to them
    #     # prepare train dataframe
    #     self.train_df = self.tabular_data[self.tabular_data['lesion_id'].isin(
    #         self.train_patients.lesion_id)].reset_index(drop=True)
    #     # prepare test dataframe
    #     self.test_df = self.tabular_data[self.tabular_data['lesion_id'].isin(
    #         self.test_patients.lesion_id)].reset_index(drop=True)
    #     # prepare val dataframe
    #     self.val_df = self.tabular_data[self.tabular_data['lesion_id'].isin(
    #         self.val_patients.lesion_id)].reset_index(drop=True)

    #     # -----------------------------------------------------------------------------------
    #     # # ONLY FOR OVERFITTING ON ONE IMAGE
    #     # # self.train_df = self.train_df.iloc[:20]
    #     # self.train_df = self.train_df.groupby(
    #     #     'label').apply(lambda x: x.sample(20)).droplevel(0).reset_index(drop=True)
    #     # self.val_df = self.val_df.groupby(
    #     #     'label').apply(lambda x: x.sample(5)).droplevel(0).reset_index(drop=True)
    #     # self.test_df = self.test_df.groupby(
    #     #     'label').apply(lambda x: x.sample(5)).droplevel(0).reset_index(drop=True)
    #     # # self.train_df = self.train_df.drop(
    #     # #     ['level_0', 'index', 'Unnamed: 0'], axis=1)
    #     # # self.val_df = self.train_df
    #     # # self.test_df = self.train_df

    #     # print('image ids in train: ', self.train_df.image_id)
    #     # print('classes in train: ', self.train_df.label)

    #     # print('image ids in val: ', self.val_df.image_id)
    #     # print('classes in val: ', self.val_df.label)

    #     print('number of patients in train: ', len(self.train_df))
    #     print('patient IDs in train: ', self.train_df.lesion_id.unique())
    #     print('number of patients in val: ', len(self.val_df))
    #     print('patient IDs in val: ', self.val_df.lesion_id.unique())

    def prepare_data(self, seed):

        # ----------------------------------------
        # NEW
        # read train and test data
        self.train_data = pd.read_csv(root_dir + train_dir)
        self.test_df = pd.read_csv(root_dir + test_dir)

        # split train data into train and val
        # ----------------------------------------
        # split the data by patient ID
        train_patient_label_list = self.train_data.groupby(
            'lesion_id')['label'].first()
        train_patient_label_df = pd.DataFrame(train_patient_label_list)
        train_patient_label_df = train_patient_label_df.reset_index()

        # get stritified split for train and val
        ss = StratifiedSampler(torch.FloatTensor(
            train_patient_label_df.label), test_size=0.2, seed=seed)
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
        self.train_df = self.train_data[self.train_data['lesion_id'].isin(
            self.train_patients.lesion_id)].reset_index(drop=True)
        # prepare val dataframe
        self.val_df = self.train_data[self.train_data['lesion_id'].isin(
            self.val_patients.lesion_id)].reset_index(drop=True)

        # # ONLY FOR OVERFITTING ON ONE IMAGE
        # self.train_df = self.train_df.groupby(
        #     'label').apply(lambda x: x.sample(5)).droplevel(0).reset_index(drop=True)
        # self.val_df = self.val_df.groupby(
        #     'label').apply(lambda x: x.sample(5)).droplevel(0).reset_index(drop=True)
        # self.test_df = self.test_df.groupby(
        #     'label').apply(lambda x: x.sample(5)).droplevel(0).reset_index(drop=True)

        print('number of patients in train: ', len(self.train_df))
        print('patient IDs in train: ', self.train_df.lesion_id.unique())
        print('number of patients in val: ', len(self.val_df))
        print('patient IDs in val: ', self.val_df.lesion_id.unique())

    def prepare_zero_shot_data(self, seed, percent):

        # read train and test data
        self.train_data = pd.read_csv(root_dir + train_dir)
        self.test_df = pd.read_csv(root_dir + test_dir)
        self.percent = percent

        # ----------------------------------------
        # split the data by patient ID
        train_patient_label_list = self.train_data.groupby(
            'lesion_id')['label'].first()
        train_patient_label_df = pd.DataFrame(train_patient_label_list)
        train_patient_label_df = train_patient_label_df.reset_index()

        # get stritified split for train and val
        ss = StratifiedSampler(torch.FloatTensor(
            train_patient_label_df.label), test_size=self.percent, seed=seed)
        train_indices, val_indices = ss.gen_sample_array()

        # store indices of train, test, valin a dictionary
        indices = {'train': val_indices,
                   'val': train_indices
                   }

        # get the patient ids using the generated indices
        self.train_patients = train_patient_label_df.iloc[indices['train']]
        self.val_patients = train_patient_label_df.iloc[indices['val']]

        # prepare the train, test, validation datasets using the subjects assigned to them
        # prepare train dataframe
        self.train_df = self.train_data[self.train_data['lesion_id'].isin(
            self.train_patients.lesion_id)].reset_index(drop=True)
        # prepare val dataframe
        self.val_df = self.train_data[self.train_data['lesion_id'].isin(
            self.val_patients.lesion_id)].reset_index(drop=True)

        # # check the number of samples in each class
        # df_check_train = self.train_df.groupby(
        #     'label').count()['lesion_id'].copy()
        # df_check_val = self.val_df.groupby('label').count()['lesion_id'].copy()
        # for i in len(df_check_train):
        #     if df_check_train.iloc[i]['lesion_id'] < 2:
        #         class_subset_val = df_check_val[df_check_val['label'] == df_check_train.iloc[i]['label']].reset_index()
        #         pos_idx = int(torch.randint(len(class_subset_val), (1,)))
        #         pass

        print('number of patients in train: ', len(self.train_df))
        print('patient IDs in train: ', self.train_df.lesion_id.unique())
        print('# samples in each class (TRAIN): ',
              self.train_df.groupby('label').count())
        print('number of patients in val: ', len(self.val_df))
        print('number of patients in val: ', len(self.val_df))
        print('patient IDs in val: ', self.val_df.lesion_id.unique())

    def set_supervised_multimodal_dataloader(self):

        # create the dataset object using the dataframes created above
        self.train = Supervised_Multimodal_Dataset(self.train_df, image_base_dir=image_dir,
                                                   target=TARGET, features=FEATURES, transform=self.get_transforms()['train'])

        self.test = Supervised_Multimodal_Dataset(self.test_df, image_base_dir=image_dir,
                                                  target=TARGET, features=FEATURES, transform=self.get_transforms()['val'])

        self.val = Supervised_Multimodal_Dataset(self.val_df, image_base_dir=image_dir,
                                                 target=TARGET, features=FEATURES, transform=self.get_transforms()['val'])

    def set_contrastive_loss_dataloader(self):

        # create the dataset object using the dataframes created above
        self.train = Supervised_Multimodal_Dataset(self.train_df, image_base_dir=image_dir,
                                                   target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms()['train'], self.n_views))

        self.test = Supervised_Multimodal_Dataset(self.test_df, image_base_dir=image_dir,
                                                  target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms()['val'], self.n_views))

        self.val = Supervised_Multimodal_Dataset(self.val_df, image_base_dir=image_dir,
                                                 target=TARGET, features=FEATURES, transform=ContrastiveLearningViewGenerator(self.get_transforms()['val'], self.n_views))

    def set_triplet_dataloader(self):

        # create the dataset object using the dataframes created above
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
        # return n positive pairs per image
        return torch.stack([self.base_transform(x) for _ in range(self.n_views)])
