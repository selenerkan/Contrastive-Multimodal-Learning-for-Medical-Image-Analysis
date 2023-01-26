# this file is generated to rearrange the folder structure after getting segmentated images of brain MRIs
import nibabel as nib
import os
import glob
import numpy as np
from monai import transforms
import math

# function to mask the hippocampus
def mask_hippocampus(orj_img, segmented_img, class_val):
    bool_array = segmented_img == class_val
    masked_img=orj_img*bool_array.squeeze()
    return masked_img

def masking():
    root_orj_img=r'/home/guests/selen_erkan/datasets/ADNI/images/preprocessed'
    root_seg_img=r'/home/guests/selen_erkan/datasets/ADNI/images/segmentation'
    dir_masked_img=r'/home/guests/selen_erkan/datasets/ADNI/images/hippocampus'

    for seg_img_folder in os.listdir(root_seg_img):

        seg_img_name= glob.glob(os.path.join(root_seg_img,seg_img_folder,'*_MALPEM.nii.gz'))
        
        # raise exception if there are more than 1 segmented images 
        if len(seg_img_name) > 1:
            raise  Exception('There are more than two segmented images in the folder: ' + str(seg_img_folder))
        
        seg_img_name = seg_img_name[0]
        img_name=seg_img_name.split('/')[-1].split('_')[0] + '.nii.gz'

        print('working on image:', img_name)
        
        # get the image paths
        path_orj_img = os.path.join(root_orj_img,img_name)
        path_seg_img = os.path.join(root_seg_img,seg_img_name)

        # read the images
        orj_image = nib.load(path_orj_img)
        orj_image=orj_image.get_fdata()
        seg_image = nib.load(path_seg_img)
        seg_image=seg_image.get_fdata()

        # mask the hippocampus
        masked_image=mask_hippocampus(orj_image, seg_image, 20)
        
        ni_masked_image = nib.Nifti1Image(masked_image, affine=np.eye(4))
        nib.save(ni_masked_image, os.path.join(dir_masked_img,img_name))

def crop_image(image, size):

    # get the mean of nonzero region
    nonzero = np.nonzero(image)
    x,y,z=math.ceil(np.unique(nonzero[0]).mean()),math.ceil(np.unique(nonzero[1]).mean()),math.ceil(np.unique(nonzero[2]).mean())

    # get the size of the nonzero element
    size_x = nonzero[0].max() - nonzero[0].min()
    size_y = nonzero[1].max() - nonzero[1].min()
    size_z = nonzero[2].max() - nonzero[2].min() 

    # get the size of the crop
    crop_size=max(size_x,size_y,size_z,size)
    
    crop=transforms.SpatialCrop(roi_center=(x,y,z), roi_size=(crop_size,crop_size,crop_size))
    cropped_img=crop(np.expand_dims(image, axis=0))
    
    return cropped_img

def cropping():
    dir_hippo=r'/home/guests/selen_erkan/datasets/ADNI/images/hippocampus'
    dir_cropped_hippo = r'/home/guests/selen_erkan/datasets/ADNI/images/cropped_hippocampus'

    for img_name in os.listdir(dir_hippo):
        img_name_path= os.path.join(dir_hippo,img_name)

        # read one image
        image = nib.load(img_name_path)
        image= image.get_fdata()

        print('IMAGE: ', img_name)
        
        cropped_img = crop_image(image,64)
        
        ni_masked_image = nib.Nifti1Image(cropped_img.numpy(), affine=np.eye(4))
        nib.loadsave.save(ni_masked_image, os.path.join(dir_cropped_hippo,img_name))

masking()
cropping()