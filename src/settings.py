import numpy as np
import os
import nibabel as nib


import torchvision.transforms as transforms

IMAGE_PATH = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\images'

CSV_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\labels'

IMAGE_SIZE = (192, 192, 160)

training_transformations = transforms.Compose([
    transforms.ToTensor()])

target_transformations = None
