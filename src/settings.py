CHECKPOINT_DIR = r'/home/guests/selen_erkan/experiments/checkpoints'
IMAGE_PATH = r"/home/guests/selen_erkan/datasets/ADNI/images/preprocessed"
CSV_FILE = r"/home/guests/selen_erkan/datasets/ADNI/tabular/adni_final.csv"

FEATURES = ['age', 'gender_numeric', 'education', 'APOE4',
            'FDG', 'AV45', 'TAU', 'PTAU', 'MMSE', 'label_numeric',
            'FDG_missing', 'TAU_missing', 'PTAU_missing', 'AV45_missing']

TARGET = 'label_numeric'
# use below code for run gender prediction
# TARGET = 'gender_numeric'

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
    'learning_rate': 0.013,
    'weight_decay': 0.0001,
    'checkpoint': None,
    'checkpoint_flag': False
}

resnet_config = {
    'batch_size': 32,
    'max_epochs': 100,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.03,
    'weight_decay': 0.0001,
    'checkpoint': None,
    'checkpoint_flag': False
}


supervised_config = {
    'batch_size': 32,
    'max_epochs': 80,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.013,
    'weight_decay': 0.01,
    'checkpoint': None,
    # 'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive/lr=0.001_wd=0_27.01.2023-17.49-epoch=079.ckpt',
    'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/triplet/lr=0.013_wd=0.01_01.02.2023-17.19-epoch=020.ckpt',
    'checkpoint_flag': False,
    'contrastive_checkpoint_flag': True
}

daft_config = {
    'batch_size': 32,
    'max_epochs': 80,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.013,
    'weight_decay': 0.01,
    'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/supervised/25.01.2023-18.49-epoch=029.ckpt',
    'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive/25.01.2023-17.14-epoch=029.ckpt',
    'checkpoint_flag': False,
    'contrastive_checkpoint_flag': False
}

contrastive_config = {
    'batch_size': 32,
    'max_epochs': 80,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.001,
    'weight_decay': 0,
    'checkpoint': None,
    'checkpoint_flag': False
}

triplet_config = {
    'batch_size': 32,
    'max_epochs': 80,
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.013,
    'weight_decay': 0,
    'checkpoint': None,
    'checkpoint_flag': False
}

knn_config = {
    'batch_size': 32,
    'model': 'contrastive',
    'age': None,
    'spatial_size': (120, 120, 120),
    'learning_rate': 0.013,
    'weight_decay': 0.01,
    'n_neighbors': 5,
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive/lr=0.001_wd=0_27.01.2023-17.49-epoch=079.ckpt',
    'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/triplet/lr=0.013_wd=0.01_01.02.2023-17.19-epoch=020.ckpt',
    
}
