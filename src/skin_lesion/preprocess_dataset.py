import os
import shutil
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


def preprocess_columns(data_dir):

    # this code is being used to encode the label in .csv file to numeric values
    # importing metadata
    metadata = pd.read_csv(data_dir + r'\HAM10000_metadata.csv')

    # label encoding the seven classes for skin cancers
    le = LabelEncoder()
    le.fit(metadata['dx'])
    LabelEncoder()
    print("Classes:", list(le.classes_))
    metadata['label'] = le.transform(metadata["dx"])

    # remoce the samples with unknown localization and sex
    metadata = metadata.drop(
        metadata[metadata['localization'] == 'unknown'].index, axis=0).reset_index(drop=True)
    metadata = metadata.drop(
        metadata[metadata['sex'] == 'unknown'].index, axis=0).reset_index(drop=True)

    # replace the empty age values with the mean
    mean = metadata[metadata['age'] > 0]['age'].mean()
    metadata.loc[(metadata['age'] == 0) | (
        metadata['age'].isna()), 'age'] = mean

    # replace categorical columns to numerical
    metadata['sex_numeric'] = pd.factorize(metadata['sex'])[0]
    metadata['localization_numeric'] = pd.factorize(
        metadata['localization'])[0]

    metadata.to_csv(data_dir + r'\HAM10000_metadata.csv')


def arrange_data_folders(data_dir, dest_dir):
    # this code is being used to segregate the images into folders of their respective classes.

    # Read the metadata file:
    metadata = pd.read_csv(data_dir + '/HAM10000_metadata.csv')
    label = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
    label_images = []

    # Copy the images into new folder structure:
    for i in label:
        os.mkdir(dest_dir + "/" + str(i) + "/")
        sample = metadata[metadata['dx'] == i]['image_id']
        label_images.extend(sample)
        for id in label_images:
            shutil.copyfile((data_dir + "/HAM10K_images/" + id + ".jpg"),
                            (dest_dir + "/" + i + "/"+id+".jpg"))
        label_images = []


def estimate_weights_mfb(data_dir, label):

    metadata = pd.read_csv(data_dir + r'\HAM10000_metadata.csv')

    class_weights = np.zeros_like(label, dtype=np.float)
    counts = np.zeros_like(label)
    for i, l in enumerate(label):
        counts[i] = metadata[metadata['dx'] == str(l)]['dx'].value_counts()[0]
    counts = counts.astype(np.float)
    median_freq = np.median(counts)
    for i, label in enumerate(label):
        class_weights[i] = median_freq / counts[i]
    return class_weights


def one_hot_encode(data_dir):

    metadata = pd.read_csv(data_dir + r'\HAM10000_metadata.csv')

    # create encoder object
    enc = OneHotEncoder(handle_unknown='ignore')
    # encode localization column
    enc.fit(metadata[['localization']])
    localization_one_hot = enc.transform(metadata[['localization']]).toarray()
    # save encodings in a pandas df
    df = pd.DataFrame(localization_one_hot, columns=enc.categories_[0])
    # combine metadata csv file with one-hot-encodings
    metadata = pd.concat([metadata, df], axis=1)

    metadata.to_csv(data_dir + r'\HAM10000_metadata.csv')


if __name__ == '__main__':

    # A path to the folder which has all the images:
    data_dir = os.getcwd() + r"\data\skin_lesion"
    # A path to the folder where you want to store the rearranged images:
    dest_dir = os.getcwd() + r"\data\skin_lesion\HAM10K_grouped_images"
    # lables in data
    label = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv',  'vasc']

    preprocess_columns(data_dir)
    arrange_data_folders(data_dir=data_dir, dest_dir=dest_dir)

    # calculate class weights
    classweight = estimate_weights_mfb(data_dir, label)
    for i in range(len(label)):
        print(label[i], ":", classweight[i])

    one_hot_encode(data_dir)
