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
# seed_list = [1997, 25]
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
            # '1997': CHECKPOINT_DIR + r'/TABULAR/03.06.2023-04.37/train/03.06.2023-04.37_ADNI_SEED=1997_lr=0.013_wd=0-epoch=039.ckpt',
            # '25': CHECKPOINT_DIR + r'/TABULAR/03.06.2023-04.53/train/03.06.2023-04.53_ADNI_SEED=25_lr=0.013_wd=0-epoch=039.ckpt',
            # '12': CHECKPOINT_DIR + r'/TABULAR/03.06.2023-05.08/train/03.06.2023-05.08_ADNI_SEED=12_lr=0.013_wd=0-epoch=039.ckpt',
            # '1966': CHECKPOINT_DIR + r'/TABULAR/03.06.2023-05.23/train/03.06.2023-05.23_ADNI_SEED=1966_lr=0.013_wd=0-epoch=039.ckpt',
            # '3297': CHECKPOINT_DIR + r'/TABULAR/03.06.2023-05.37/train/03.06.2023-05.37_ADNI_SEED=3297_lr=0.013_wd=0-epoch=039.ckpt'
        },
    },

    'resnet_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,  # 0.03,
        'weight_decay': 0,  # 0.0001,
        'checkpoint': {
            # '1997': CHECKPOINT_DIR + r'/RESNET/03.06.2023-01.30/train/03.06.2023-01.30_ADNI_SEED=1997_lr=0.013_wd=0-epoch=032.ckpt',
            # '25': CHECKPOINT_DIR + r'/RESNET/03.06.2023-01.49/train/03.06.2023-01.49_ADNI_SEED=25_lr=0.013_wd=0-epoch=032.ckpt',
            # '12': CHECKPOINT_DIR + r'/RESNET/03.06.2023-02.21/train/03.06.2023-02.21_ADNI_SEED=12_lr=0.013_wd=0-epoch=032.ckpt',
            # '1966': CHECKPOINT_DIR + r'/RESNET/03.06.2023-02.37/train/03.06.2023-02.37_ADNI_SEED=1966_lr=0.013_wd=0-epoch=032.ckpt',
            # '3297': CHECKPOINT_DIR + r'/RESNET/03.06.2023-02.54/train/03.06.2023-02.54_ADNI_SEED=3297_lr=0.013_wd=0-epoch=032.ckpt'
        },
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
        'correlation': False,
        # 'checkpoint_concat': {
        #     '1997': CHECKPOINT_DIR + r'/_SUPERVISED/CONCAT/02.06.2023-22.26/train/02.06.2023-22.26_ADNI_SEED=1997_lr=0.013_wd=0-epoch=034.ckpt',
        #     '25': CHECKPOINT_DIR + r'/_SUPERVISED/CONCAT/02.06.2023-22.43/train/02.06.2023-22.43_ADNI_SEED=25_lr=0.013_wd=0-epoch=034.ckpt',
        #     '12': CHECKPOINT_DIR + r'/_SUPERVISED/CONCAT/02.06.2023-22.59/train/02.06.2023-22.59_ADNI_SEED=12_lr=0.013_wd=0-epoch=034.ckpt',
        #     '1966': CHECKPOINT_DIR + r'/_SUPERVISED/CONCAT/02.06.2023-23.17/train/02.06.2023-23.17_ADNI_SEED=1966_lr=0.013_wd=0-epoch=034.ckpt',
        #     '3297': CHECKPOINT_DIR + r'/_SUPERVISED/CONCAT/02.06.2023-23.36/train/02.06.2023-23.36_ADNI_SEED=3297_lr=0.013_wd=0-epoch=034.ckpt'},
        # 'checkpoint_corr': {
        #     '1997': CHECKPOINT_DIR + r'/_SUPERVISED/CORRELATION/03.06.2023-04.39/train/03.06.2023-04.39_ADNI_SEED=1997_lr=0.013_wd=0-epoch=039.ckpt',
        #     '25': CHECKPOINT_DIR + r'/_SUPERVISED/CORRELATION/03.06.2023-04.57/train/03.06.2023-04.57_ADNI_SEED=25_lr=0.013_wd=0-epoch=039.ckpt',
        #     '12': CHECKPOINT_DIR + r'/_SUPERVISED/CORRELATION/03.06.2023-05.12/train/03.06.2023-05.12_ADNI_SEED=12_lr=0.013_wd=0-epoch=039.ckpt',
        #     '1966': CHECKPOINT_DIR + r'/_SUPERVISED/CORRELATION/03.06.2023-05.28/train/03.06.2023-05.28_ADNI_SEED=1966_lr=0.013_wd=0-epoch=039.ckpt',
        #     '3297': CHECKPOINT_DIR + r'/_SUPERVISED/CORRELATION/03.06.2023-05.44/train/03.06.2023-05.44_ADNI_SEED=3297_lr=0.013_wd=0-epoch=039.ckpt'},
    },

    'daft_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,
        'weight_decay': 0.01,
        'checkpoint': {
            # '1997': CHECKPOINT_DIR + r'/DAFT/train/02.06.2023-14.36_ADNI_SEED=1997_lr=0.013_wd=0.01-epoch=039.ckpt',
            # '25': CHECKPOINT_DIR + r'/DAFT/train/02.06.2023-14.50_ADNI_SEED=25_lr=0.013_wd=0.01-epoch=039.ckpt',
            # '12': CHECKPOINT_DIR + r'/DAFT/train/02.06.2023-15.05_ADNI_SEED=12_lr=0.013_wd=0.01-epoch=039.ckpt',
            # '1966': CHECKPOINT_DIR + r'/DAFT/train/02.06.2023-15.20_ADNI_SEED=1966_lr=0.013_wd=0.01-epoch=039.ckpt',
            # '3297': CHECKPOINT_DIR + r'/DAFT/train/02.06.2023-15.38_ADNI_SEED=3297_lr=0.013_wd=0.01-epoch=039.ckpt'
        },
    },

    'film_config': {
        'batch_size': 512,
        'max_epochs': 40,
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.013,
        'weight_decay': 0,
        'checkpoint': {
            # '1997': CHECKPOINT_DIR + r'/FILM/train/02.06.2023-14.37_ADNI_SEED=1997_lr=0.013_wd=0-epoch=024.ckpt',
            # '25': CHECKPOINT_DIR + r'/FILM/train/02.06.2023-14.52_ADNI_SEED=25_lr=0.013_wd=0-epoch=024.ckpt',
            # '12': CHECKPOINT_DIR + r'/FILM/train/02.06.2023-15.07_ADNI_SEED=12_lr=0.013_wd=0-epoch=024.ckpt',
            # '1966': CHECKPOINT_DIR + r'/FILM/train/02.06.2023-15.22_ADNI_SEED=1966_lr=0.013_wd=0-epoch=024.ckpt',
            # '3297': CHECKPOINT_DIR + r'/FILM/train/02.06.2023-15.44_ADNI_SEED=3297_lr=0.013_wd=0-epoch=024.ckpt'
        },
    },

    'triplet_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 0.01,
        'weight_decay': 0,
        'spatial_size': (64, 64, 64),
        'alpha_center': 0.01,
        'alpha_triplet': 0.2,
        'correlation': False,
        'checkpoint': None,
        # 'checkpoint_concat': {
        #     '1997': CHECKPOINT_DIR + r'/TRIPLET/CONCAT/04.06.2023-13.05/train/04.06.2023-13.05_ADNI_SEED=1997_lr=0.01_wd=0-epoch=034.ckpt',
        #     '25': CHECKPOINT_DIR + r'/TRIPLET/CONCAT/04.06.2023-13.52/train/04.06.2023-13.52_ADNI_SEED=25_lr=0.01_wd=0-epoch=034.ckpt',
        #     '12': CHECKPOINT_DIR + r'/TRIPLET/CONCAT/04.06.2023-14.38/train/04.06.2023-14.38_ADNI_SEED=12_lr=0.01_wd=0-epoch=034.ckpt',
        #     '1966': CHECKPOINT_DIR + r'/TRIPLET/CONCAT/04.06.2023-15.25/train/04.06.2023-15.25_ADNI_SEED=1966_lr=0.01_wd=0-epoch=034.ckpt',
        #     '3297': CHECKPOINT_DIR + r'/TRIPLET/CONCAT/04.06.2023-16.12/train/04.06.2023-16.12_ADNI_SEED=3297_lr=0.01_wd=0-epoch=034.ckpt'},
        # 'checkpoint_corr': {
        #     '1997': CHECKPOINT_DIR + r'/TRIPLET/02.06.2023-23.00/train/02.06.2023-23.00_ADNI_SEED=1997_lr=0.01_wd=0-epoch=039.ckpt',
        #     '25': CHECKPOINT_DIR + r'/TRIPLET/02.06.2023-23.53/train/02.06.2023-23.53_ADNI_SEED=25_lr=0.01_wd=0-epoch=039.ckpt',
        #     '12': CHECKPOINT_DIR + r'/TRIPLET/03.06.2023-00.43/train/03.06.2023-00.43_ADNI_SEED=12_lr=0.01_wd=0-epoch=039.ckpt',
        #     '1966': CHECKPOINT_DIR + r'/TRIPLET/03.06.2023-01.30/train/03.06.2023-01.30_ADNI_SEED=1966_lr=0.01_wd=0-epoch=039.ckpt',
        #     '3297': CHECKPOINT_DIR + r'/TRIPLET/03.06.2023-02.35/train/03.06.2023-02.35_ADNI_SEED=3297_lr=0.01_wd=0-epoch=039.ckpt'},
    },

    'modality_specific_center_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.01,
        'weight_decay': 0,
        'checkpoint': {
            # '1997': CHECKPOINT_DIR + r'/MODALITY_SPECIFIC/04.06.2023-13.37/train/04.06.2023-13.37_ADNI_SEED=1997_lr=0.01_wd=0-epoch=026.ckpt',
            # '25': CHECKPOINT_DIR + r'/MODALITY_SPECIFIC/04.06.2023-13.53/train/04.06.2023-13.53_ADNI_SEED=25_lr=0.01_wd=0-epoch=026.ckpt',
            # '12': CHECKPOINT_DIR + r'/MODALITY_SPECIFIC/04.06.2023-14.08/train/04.06.2023-14.08_ADNI_SEED=12_lr=0.01_wd=0-epoch=026.ckpt',
            # '1966': CHECKPOINT_DIR + r'/MODALITY_SPECIFIC/04.06.2023-14.24/train/04.06.2023-14.24_ADNI_SEED=1966_lr=0.01_wd=0-epoch=026.ckpt',
            # '3297': CHECKPOINT_DIR + r'/MODALITY_SPECIFIC/04.06.2023-14.40/train/04.06.2023-14.40_ADNI_SEED=3297_lr=0.01_wd=0-epoch=026.ckpt',
        },
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
            # '1997': CHECKPOINT_DIR + r'/CROSS_MODAL/04.06.2023-16.44/train/04.06.2023-16.44_ADNI_SEED=1997_lr=0.01_wd=0-epoch=025.ckpt',
            # '25': CHECKPOINT_DIR + r'/CROSS_MODAL/04.06.2023-17.00/train/04.06.2023-17.00_ADNI_SEED=25_lr=0.01_wd=0-epoch=025.ckpt',
            # '12': CHECKPOINT_DIR + r'/CROSS_MODAL/04.06.2023-17.17/train/04.06.2023-17.17_ADNI_SEED=12_lr=0.01_wd=0-epoch=025.ckpt',
            # '1966': CHECKPOINT_DIR + r'/CROSS_MODAL/04.06.2023-17.35/train/04.06.2023-17.35_ADNI_SEED=1966_lr=0.01_wd=0-epoch=025.ckpt',
            # '3297': CHECKPOINT_DIR + r'/CROSS_MODAL/04.06.2023-17.51/train/04.06.2023-17.51_ADNI_SEED=3297_lr=0.01_wd=0-epoch=025.ckpt',
        },
        'alpha_center': 0.01,
    },

    'center_loss_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'spatial_size': (64, 64, 64),
        'learning_rate': 0.01,
        'weight_decay': 0,
        'correlation': True,
        'checkpoint': None,
        # 'checkpoint_concat': {
        #     '1997': CHECKPOINT_DIR + r'/CENTER_LOSS/CONCAT/04.06.2023-17.07/train/04.06.2023-17.07_ADNI_SEED=1997_lr=0.01_wd=0-epoch=028.ckpt',
        #     '25': CHECKPOINT_DIR + r'/CENTER_LOSS/CONCAT/04.06.2023-17.26/train/04.06.2023-17.26_ADNI_SEED=25_lr=0.01_wd=0-epoch=028.ckpt',
        #     '12': CHECKPOINT_DIR + r'/CENTER_LOSS/CONCAT/04.06.2023-17.43/train/04.06.2023-17.43_ADNI_SEED=12_lr=0.01_wd=0-epoch=028.ckpt',
        #     '1966': CHECKPOINT_DIR + r'/CENTER_LOSS/CONCAT/04.06.2023-17.59/train/04.06.2023-17.59_ADNI_SEED=1966_lr=0.01_wd=0-epoch=028.ckpt',
        #     '3297': CHECKPOINT_DIR + r'/CENTER_LOSS/CONCAT/04.06.2023-18.14/train/04.06.2023-18.14_ADNI_SEED=3297_lr=0.01_wd=0-epoch=028.ckpt'},
        # 'checkpoint_corr': {
        #     '1997': CHECKPOINT_DIR + r'/CENTER_LOSS/CORRELATION/04.06.2023-18.07/train/04.06.2023-18.07_ADNI_SEED=1997_lr=0.01_wd=0-epoch=031.ckpt',
        #     '25': CHECKPOINT_DIR + r'/CENTER_LOSS/CORRELATION/04.06.2023-18.22/train/04.06.2023-18.22_ADNI_SEED=25_lr=0.01_wd=0-epoch=031.ckpt',
        #     '12': CHECKPOINT_DIR + r'/CENTER_LOSS/CORRELATION/04.06.2023-18.38/train/04.06.2023-18.38_ADNI_SEED=12_lr=0.01_wd=0-epoch=031.ckpt',
        #     '1966': CHECKPOINT_DIR + r'/CENTER_LOSS/CORRELATION/04.06.2023-18.55/train/04.06.2023-18.55_ADNI_SEED=1966_lr=0.01_wd=0-epoch=031.ckpt',
        #     '3297': CHECKPOINT_DIR + r'/CENTER_LOSS/CORRELATION/04.06.2023-19.12/train/04.06.2023-19.12_ADNI_SEED=3297_lr=0.01_wd=0-epoch=031.ckpt'},
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
