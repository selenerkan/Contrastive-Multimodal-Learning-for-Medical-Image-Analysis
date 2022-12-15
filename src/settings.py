import torchvision.transforms as transforms

IMAGE_PATH = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\images\preprocessed\Combined'
CSV_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\labels\tabular_image_labels'
TABULAR_DATA_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\tabular'

FEATURES = ['age', 'gender_numeric', 'education', 'APOE4',
            'FDG', 'AV45', 'TAU', 'PTAU', 'MMSE', 'label_numeric',
            'FDG_missing', 'TAU_missing', 'PTAU_missing', 'AV45_missing']

TARGET = 'label_numeric'

IMAGE_SIZE = (182, 218, 182)
# image_spacing = (1.0, 1.0, 1.0)
# image_origin = (-90.0, 126.0, -72.0)

TRAIN_SIZE = 1
VAL_SIZE = 1
TEST_SIZE = 1

# transformation for the input images
transformation = transforms.Compose([
    transforms.ToTensor(),
])

# transformation for the labels
target_transformations = None
