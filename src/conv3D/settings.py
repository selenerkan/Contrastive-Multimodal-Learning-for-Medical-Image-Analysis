import torchvision.transforms as transforms

IMAGE_PATH = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\images'

CSV_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\labels'

IMAGE_SIZE = (256, 256, 170)

TRAIN_SIZE = 1
VAL_SIZE = 3
TEST_SIZE = 1

# transformation for the input images
transformation = transforms.Compose([
    transforms.ToTensor(),
])

# transformation for the labels
target_transformations = None
