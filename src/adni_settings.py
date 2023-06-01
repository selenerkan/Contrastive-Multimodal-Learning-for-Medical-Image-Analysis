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
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.001,  # 0.013,
        'weight_decay': 0,  # 0.0001,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r''},
    },

    'resnet_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.01,  # 0.03,
        'weight_decay': 0,  # 0.0001,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r''},
    },


    'supervised_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'checkpoint': None,
        # 'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive/lr=0.001_wd=0_27.01.2023-17.49-epoch=079.ckpt',
        # 'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/triplet/lr=0.013_wd=0.01_01.02.2023-17.19-epoch=020.ckpt',
        'correlation': False,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r''},
    },

    'daft_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,
        'weight_decay': 0.01,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r''},
    },

    'film_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r''},
    },

    'triplet_center_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 0.001,
        'weight_decay': 0,
        'alpha_center': 0.01,
        'alpha_triplet': 0.2,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r''},
    },

    'contrastive_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.001,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r''},
    },

    'triplet_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.001,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r''},
    },
}
