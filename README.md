# Multimodal Self Supervised Network for ADNI

## Purpose

The purpose of this project is to combine imaging and non-imaging dataset to make better disease diagnostics.

## 1. Dataset

### 1.1. ADNI

For initial evaluation of the approach ADNI dataset is used. The dataset can be retrieved from: https://ida.loni.usc.edu/login.jsp?project=ADNI after getting access rights.

There is one image dataset and one tabular dataset. Both dataset consist inputs of the patients. There are 299 different patients with multiple visits. On average one patient has 2-3 visits. After each visit an MRI scan is taken and a new input to the tabular data entered. So, both the tabular and image dataset has more than one input for patients.

### 1.2. HAM10K

HAM10K skin lesion dataset is also experimented to provide more results on the research. The dataset can be found at: https://github.com/ptschandl/HAM10000_dataset

The dataset contains both image and tabular modalities. There are around 1700 different patients with multiple visits. On average one patient has 1.3 visits. After each visit a 2D skin  lesion scan is taken and a new input to the tabular data entered.

## 2. Preprocessing

### 2.1 ADNI Preprocessing

The following preprocessing used for all the MRI images: https://github.com/quqixun/BrainPrep

The implementation includes the following preprocessing steps:

- registration
- skull stripping
- bias field correction

After preprocessing every image has the shape of (64, 64, 64)

### 2.2 Tabular data Preprocessing

For the tabular modalities of the datasets missing values are filled with the average value of the corresponding patient. 
If the patient has only null values for a specific feature, these values are filled with the average value of the column.
Patients having too many null values in various features are removed


## 3. Model

There are three baseline models implemented for the priject.

1. 3D Convolutional Network (only for image data)
2. ResNet Network (only for image data)
3. Multimodal Network (for both Image and tabular data)

### 3.1 Convolutional Network

### 3.2 ResNet Network

### 3.3 Multimodal Network

Multimodal Network takes both an image and a tabular data input. Image data is feed to the same ResNet Model used in 2.2. Image representations that are retrieved from this block concatenated to the related tabular feature vector.
