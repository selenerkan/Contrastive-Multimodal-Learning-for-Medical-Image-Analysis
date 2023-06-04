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
        'learning_rate': 0.013,  # 0.013,
        'weight_decay': 0,  # 0.0001,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'/TABULAR/train/01.06.2023-12.50_ADNI_SEED=1997_lr=0.001_wd=0-epoch=039.ckpt',
            '25': CHECKPOINT_DIR + r'/TABULAR/train/01.06.2023-13.05_ADNI_SEED=25_lr=0.001_wd=0-epoch=039.ckpt',
            '12': CHECKPOINT_DIR + r'/TABULAR/train/01.06.2023-13.21_ADNI_SEED=12_lr=0.001_wd=0-epoch=039.ckpt',
            '1966': CHECKPOINT_DIR + r'/TABULAR/train/01.06.2023-13.36_ADNI_SEED=1966_lr=0.001_wd=0-epoch=039.ckpt',
            '3297': CHECKPOINT_DIR + r'/TABULAR/train/01.06.2023-13.50_ADNI_SEED=3297_lr=0.001_wd=0-epoch=039.ckpt'},
    },

    'resnet_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,  # 0.03,
        'weight_decay': 0,  # 0.0001,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'/RESNET/train/01.06.2023-12.16_ADNI_SEED=1997_lr=0.03_wd=0-epoch=039.ckpt',
            '25': CHECKPOINT_DIR + r'/RESNET/train/01.06.2023-12.33_ADNI_SEED=25_lr=0.03_wd=0-epoch=039.ckpt',
            '12': CHECKPOINT_DIR + r'/RESNET/train/01.06.2023-12.49_ADNI_SEED=12_lr=0.03_wd=0-epoch=039.ckpt',
            '1966': CHECKPOINT_DIR + r'/RESNET/train/01.06.2023-13.06_ADNI_SEED=1966_lr=0.03_wd=0-epoch=039.ckpt',
            '3297': CHECKPOINT_DIR + r'/RESNET/train/01.06.2023-13.25_ADNI_SEED=3297_lr=0.03_wd=0-epoch=039.ckpt'},
    },


    'supervised_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,
        'weight_decay': 0,
        'checkpoint': None,
        # 'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive/lr=0.001_wd=0_27.01.2023-17.49-epoch=079.ckpt',
        # 'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/triplet/lr=0.013_wd=0.01_01.02.2023-17.19-epoch=020.ckpt',
        'correlation': True,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'/_SUPERVISED/CONCAT/train/01.06.2023-12.47_ADNI_SEED=1997_lr=0.001_wd=0.01-epoch=039.ckpt',
            '25': CHECKPOINT_DIR + r'/_SUPERVISED/CONCAT/train/01.06.2023-13.05_ADNI_SEED=25_lr=0.001_wd=0.01-epoch=039.ckpt',
            '12': CHECKPOINT_DIR + r'/_SUPERVISED/CONCAT/train/01.06.2023-13.24_ADNI_SEED=12_lr=0.001_wd=0.01-epoch=039.ckpt',
            '1966': CHECKPOINT_DIR + r'/_SUPERVISED/CONCAT/train/01.06.2023-13.43_ADNI_SEED=1966_lr=0.001_wd=0.01-epoch=039.ckpt',
            '3297': CHECKPOINT_DIR + r'/_SUPERVISED/CONCAT/train/01.06.2023-14.02_ADNI_SEED=3297_lr=0.001_wd=0.01-epoch=039.ckpt'},
    },

    'daft_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,
        'weight_decay': 0.01,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'/DAFT/train/01.06.2023-13.49_ADNI_SEED=1997_lr=0.013_wd=0.01-epoch=035.ckpt',
            '25': CHECKPOINT_DIR + r'/DAFT/train/01.06.2023-14.13_ADNI_SEED=25_lr=0.013_wd=0.01-epoch=035.ckpt',
            '12': CHECKPOINT_DIR + r'/DAFT/train/01.06.2023-14.33_ADNI_SEED=12_lr=0.013_wd=0.01-epoch=035.ckpt',
            '1966': CHECKPOINT_DIR + r'/DAFT/train/01.06.2023-14.50_ADNI_SEED=1966_lr=0.013_wd=0.01-epoch=035.ckpt',
            '3297': CHECKPOINT_DIR + r'/DAFT/train/01.06.2023-15.06_ADNI_SEED=3297_lr=0.013_wd=0.01-epoch=035.ckpt'},
    },

    'film_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'/FILM/train/01.06.2023-14.11_ADNI_SEED=1997_lr=0.013_wd=0-epoch=027.ckpt',
            '25': CHECKPOINT_DIR + r'/FILM/train/01.06.2023-14.31_ADNI_SEED=25_lr=0.013_wd=0-epoch=027.ckpt',
            '12': CHECKPOINT_DIR + r'/FILM/train/01.06.2023-14.49_ADNI_SEED=12_lr=0.013_wd=0-epoch=027.ckpt',
            '1966': CHECKPOINT_DIR + r'/FILM/train/01.06.2023-15.07_ADNI_SEED=1966_lr=0.013_wd=0-epoch=027.ckpt',
            '3297': CHECKPOINT_DIR + r'/FILM/train/01.06.2023-15.26_ADNI_SEED=3297_lr=0.013_wd=0-epoch=027.ckpt'},
    },

    'triplet_center_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 0.01,
        'weight_decay': 0,
        'spatial_size': (64, 64, 64),
        'alpha_center': 0.01,
        'alpha_triplet': 0.2,
        'correlation': False,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r''},
    },

    'modality_specific_center_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.01,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r'', },
        'alpha_center': 0.01,
    },

    'cross_modal_center_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.01,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r'', },
        'alpha_center': 0.01,
    },

    'center_loss_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.01,
        'weight_decay': 0,
        'correlation': False,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'',
            '25': CHECKPOINT_DIR + r'',
            '12': CHECKPOINT_DIR + r'',
            '1966': CHECKPOINT_DIR + r'',
            '3297': CHECKPOINT_DIR + r'', },
        'alpha_center': 0.01,
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

}
