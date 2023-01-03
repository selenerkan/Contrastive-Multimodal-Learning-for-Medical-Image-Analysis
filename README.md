# Multimodal Self Supervised Network for ADNI

## Purpose

The purpose of this project is to combine imaging and non-imaging dataset to make better disease diagnostics.

## 1. Dataset

For initial evaluation of the approach ADNI dataset is used. The dataset can be retrieved from: https://ida.loni.usc.edu/login.jsp?project=ADNI after getting access rights.

There is one image dataset and one tabular dataset. Both dataset consist inputs of the patients. There are 299 different patients with multiple visits. On average one patient has 2-3 visits. After each visit an MRI scan is taken and a new input to the tabular data entered. So, both the tabular and image dataset has more than one input for patients.

## 2. Preprocessing

### 2.1 Image Preprocessing

The following preprocessing used for all the MRI images: https://github.com/quqixun/BrainPrep

The implementation includes the following preprocessing steps:

- registration
- skull stripping
- bias field correction

After preprocessing every image has the shape of (182, 218, 182)

## 3. Model

There are three baseline models implemented for the priject.

1. 3D Convolutional Network (only for image data)
2. ResNet Network (only for image data)
3. Multimodal Network (for both Image and tabular data)

### 3.1 Convolutional Network

### 3.2 ResNet Network

### 3.3 Multimodal Network

Multimodal Network takes both an image and a tabular data input. Image data is feed to the same ResNet Model used in 2.2. Image representations that are retrieved from this block concatenated to the related tabular feature vector.
