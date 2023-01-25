TABULAR_DATA_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\tabular'
CHECKPOINT_DIR = r'/home/guests/selen_erkan/experiments/checkpoints'
IMAGE_PATH = r"/home/guests/selen_erkan/datasets/ADNI/images/preprocessed"
CSV_FILE = r"/home/guests/selen_erkan/datasets/ADNI/tabular/adni_final.csv"

FEATURES = ['age', 'gender_numeric', 'education', 'APOE4',
            'FDG', 'AV45', 'TAU', 'PTAU', 'MMSE', 'label_numeric',
            'FDG_missing', 'TAU_missing', 'PTAU_missing', 'AV45_missing']

TARGET = 'label_numeric'

IMAGE_SIZE = (182, 218, 182)

SEED = 473

TRAIN_SIZE = 1
VAL_SIZE = 1
TEST_SIZE = 1

tabular_config = {
    'batch_size': 32,
    'max_epochs': 100,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.0055,
    'weight_decay': 0,
}

resnet_config = {
    'batch_size': 32,
    'max_epochs': 100,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.013,
    'weight_decay': 0.01,
}


supervised_config = {
    'batch_size': 32,
    'max_epochs': 30,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.013,
    'weight_decay': 0.01,
}

contrastive_config = {
    'batch_size': 32,
    'max_epochs': 30,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.013,
    'weight_decay': 0.01,
}
