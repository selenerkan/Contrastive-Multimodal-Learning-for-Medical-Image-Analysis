import torchvision.transforms as transforms

# IMAGE_PATH = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\images\init_test_images'
IMAGE_PATH = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\images\nifti_images_tabular'

# CSV_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\labels'
CSV_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\labels\tabular_image_labels'

TABULAR_DATA_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\tabular\adni_tabular_images'

FEATURES = ['age', 'gender_encoded', 'education', 'APOE4',
            'FDG', 'AV45', 'TAU', 'PTAU', 'MMSE', 'label_encoded']

TARGET = 'label_encoded'

IMAGE_SIZE = (256, 256, 170)

TRAIN_SIZE = 1
VAL_SIZE = 1
TEST_SIZE = 1

# transformation for the input images
transformation = transforms.Compose([
    transforms.ToTensor(),
])

# transformation for the labels
target_transformations = None
