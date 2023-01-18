IMAGE_PATH = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\images\preprocessed'
CSV_FILE = r"C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\tabular\adni_final.csv"
TABULAR_DATA_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\tabular'
CHECKPOINT_DIR = r'C:\Users\Selen\Desktop\LMU\multimodal_network\checkpoints'

FEATURES = ['age', 'gender_numeric', 'education', 'APOE4',
            'FDG', 'AV45', 'TAU', 'PTAU', 'MMSE', 'label_numeric',
            'FDG_missing', 'TAU_missing', 'PTAU_missing', 'AV45_missing']

TARGET = 'label_numeric'

IMAGE_SIZE = (182, 218, 182)

SEED = 473

TRAIN_SIZE = 1
VAL_SIZE = 1
TEST_SIZE = 1

resnet_config = {
    'batch_size': 16,
    'max_epochs': 100,
    'epochs': 5,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.01,
    'weight_decay': 1e-4,
}


supervised_config = {
    'batch_size': 16,
    'max_epochs': 100,
    'epochs': 5,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
}

contrastive_config = {
    'batch_size': 8,
    'max_epochs': 100,
    'epochs': 5,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.0001,
    'weight_decay': 1e-4,
}
