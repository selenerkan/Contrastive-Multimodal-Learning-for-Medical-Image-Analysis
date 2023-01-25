# this file is generated to rearrange the folder structure after getting segmentated images of brain MRIs
import nibabel as nib
import os
import glob
import numpy as np

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

masking()