CHECKPOINT_DIR = r'/vol/aimspace/users/erks/experiments/adni_checkpoints'
image_dir = r"/vol/aimspace/users/erks/datasets/adni_full/adni_selen/images"
root_dir = r"/vol/aimspace/users/erks/datasets/adni_full/adni_selen/"
train_dir = r'/train_data.csv'
test_dir = r'/test_data.csv'
CSV_DIR = r"/vol/aimspace/users/erks/datasets/adni_full/adni_selen/adni_final.csv"

FEATURES = ['age', 'gender_numeric', 'education', 'APOE4',
            'FDG', 'TAU', 'PTAU', 'MMSE', 'label_numeric',
            'FDG_missing', 'TAU_missing', 'PTAU_missing']

TARGET = 'label_numeric'

IMAGE_SIZE = (64, 64, 64)

seed_list = [1997, 25, 12, 1966, 3297]
SEED = 473

config = {
    'tabular_config': {
        'batch_size': 32,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.001,  # 0.013,
        'weight_decay': 0.01,  # 0.0001,
        'checkpoint': None,
        'checkpoint_flag': False
    },

    'resnet_config': {
        'batch_size': 32,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.001,  # 0.03,
        'weight_decay': 0.01,  # 0.0001,
        'checkpoint': None,
        'checkpoint_flag': False
    },


    'supervised_config': {
        'batch_size': 32,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'checkpoint': None,
        # 'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive/lr=0.001_wd=0_27.01.2023-17.49-epoch=079.ckpt',
        'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/triplet/lr=0.013_wd=0.01_01.02.2023-17.19-epoch=020.ckpt',
        'checkpoint_flag': False,
        'contrastive_checkpoint_flag': False,
        'correlation':False
    },

    'daft_config': {
        'batch_size': 32,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,
        'weight_decay': 0.01,
        'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/supervised/25.01.2023-18.49-epoch=029.ckpt',
        'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive/25.01.2023-17.14-epoch=029.ckpt',
        'checkpoint_flag': False,
        'contrastive_checkpoint_flag': False
    },

    'contrastive_config': {
        'batch_size': 32,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.001,
        'weight_decay': 0,
        'checkpoint': None,
        'checkpoint_flag': False
    },

    'triplet_config': {
        'batch_size': 32,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,
        'weight_decay': 0,
        'checkpoint': None,
        'checkpoint_flag': False
    },

    'knn_config': {
        'batch_size': 32,
        'model': 'contrastive',
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,
        'weight_decay': 0.01,
        'n_neighbors': 5,
        # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive/lr=0.001_wd=0_27.01.2023-17.49-epoch=079.ckpt',
        'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/triplet/lr=0.013_wd=0.01_01.02.2023-17.19-epoch=020.ckpt',

    }

}
