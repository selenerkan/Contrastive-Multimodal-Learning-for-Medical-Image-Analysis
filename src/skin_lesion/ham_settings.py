import torch

root_dir = r'/vol/aimspace/users/erks/datasets/skin_lesion'
image_dir = root_dir + r"/HAM10K_grouped_images"
csv_dir = root_dir + r"/HAM10000_metadata.csv"
train_dir = r'/train_data.csv'
test_dir = r'/test_data.csv'
CHECKPOINT_DIR = r'/vol/aimspace/users/erks/experiments/checkpoints/'

FEATURES = ['age', 'sex_numeric', 'label', 'abdomen', 'acral',	'back',	'chest', 'ear',	'face',	'foot',
            'genital',	'hand',	'lower extremity',	'neck',	'scalp',	'trunk',	'upper extremity']
# FEATURES = ['age', 'sex_numeric', 'label', 'localization_numeric']

TARGET = 'label'

seed_list = [1997, 25, 12, 1966, 3297]
SEED = 25

image_shape = (3, 224, 224)

config = {
    'triplet_center_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'alpha_center': 0.01,
        'alpha_triplet': 0.2,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'triplet_center_cross/training/25.05.2023-16.52HAM_SEED=1997_lr=0.0001_wd=0-epoch=039.ckpt',
            '25': CHECKPOINT_DIR + r'triplet_center_cross/training/25.05.2023-17.39HAM_SEED=25_lr=0.0001_wd=0-epoch=039.ckpt',
            '12': CHECKPOINT_DIR + r'triplet_center_cross/training/25.05.2023-18.25HAM_SEED=12_lr=0.0001_wd=0-epoch=039.ckpt',
            '1966': CHECKPOINT_DIR + r'triplet_center_cross/training/25.05.2023-19.11HAM_SEED=1966_lr=0.0001_wd=0-epoch=039.ckpt',
            '3297': CHECKPOINT_DIR + r'triplet_center_cross/training/25.05.2023-19.58HAM_SEED=3297_lr=0.0001_wd=0-epoch=039.ckpt'},
    },

    'daft_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'_DAFT/training/26.05.2023-19.27_HAM_SEED=1997_lr=0.0001_wd=0-epoch=006.ckpt',
            '25': CHECKPOINT_DIR + r'_DAFT/training/26.05.2023-19.46_HAM_SEED=25_lr=0.0001_wd=0-epoch=006.ckpt',
            '12': CHECKPOINT_DIR + r'_DAFT/training/26.05.2023-20.04_HAM_SEED=12_lr=0.0001_wd=0-epoch=006.ckpt',
            '1966': CHECKPOINT_DIR + r'_DAFT/training/26.05.2023-20.21_HAM_SEED=1966_lr=0.0001_wd=0-epoch=012.ckpt',
            '3297': CHECKPOINT_DIR + r'_DAFT/training/26.05.2023-20.39_HAM_SEED=3297_lr=0.0001_wd=0-epoch=006.ckpt'},
    },


    'film_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 1e-4,  # 3e-4
        'weight_decay': 0,  # 1e-5
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'_FILM/train/26.05.2023-21.00_HAM_SEED=1997_lr=0.0001_wd=0-epoch=012.ckpt',
            '25': CHECKPOINT_DIR + r'_FILM/train/26.05.2023-21.17_HAM_SEED=25_lr=0.0001_wd=0-epoch=012.ckpt',
            '12': CHECKPOINT_DIR + r'_FILM/train/26.05.2023-21.35_HAM_SEED=12_lr=0.0001_wd=0-epoch=012.ckpt',
            '1966': CHECKPOINT_DIR + r'_FILM/train/26.05.2023-21.53_HAM_SEED=1966_lr=0.0001_wd=0-epoch=012.ckpt',
            '3297': CHECKPOINT_DIR + r'_FILM/train/30.05.2023-10.56_HAM_SEED=3297_lr=0.0001_wd=0-epoch=012.ckpt'},
    },

    'multiloss_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'alpha_center': 0.01,
        'dropout': 0,
        'correlation': False,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'CENTER_CROSS_ENT/train/CONCAT/27.05.2023-23.29_HAM_SEED=1997_lr=0.0001_wd=0-epoch=039.ckpt',
            '25': CHECKPOINT_DIR + r'CENTER_CROSS_ENT/train/CONCAT/27.05.2023-23.52_HAM_SEED=25_lr=0.0001_wd=0-epoch=039.ckpt',
            '12': CHECKPOINT_DIR + r'CENTER_CROSS_ENT/train/CONCAT/28.05.2023-00.15_HAM_SEED=12_lr=0.0001_wd=0-epoch=039.ckpt',
            '1966': CHECKPOINT_DIR + r'CENTER_CROSS_ENT/train/CONCAT/28.05.2023-00.37_HAM_SEED=1966_lr=0.0001_wd=0-epoch=039.ckpt',
            '3297': CHECKPOINT_DIR + r'CENTER_CROSS_ENT/train/CONCAT/28.05.2023-01.00_HAM_SEED=3297_lr=0.0001_wd=0-epoch=039.ckpt'},
    },

    'contrastive_pretrain_config': {
        'batch_size': 512,  # 512
        'max_epochs': 100,  # 40
        'age': None,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'checkpoint': {},
        'correlation': True,
    },

    'contrastive_center_cross_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'checkpoint': {},
        'alpha_center': 0.01,
        'alpha_contrastive': 0.2,
        'correlation': True,
    },

    'modality_center_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'final/modality_center/min_loss/09.03.2023-18.51HAM_SEED=1997_lr=0.0001_wd=0-epoch=039.ckpt',
            '25': CHECKPOINT_DIR + r'final/modality_center/min_loss/09.03.2023-19.10HAM_SEED=25_lr=0.0001_wd=0-epoch=039.ckpt',
            '12': CHECKPOINT_DIR + r'final/modality_center/min_loss/09.03.2023-18.52HAM_SEED=12_lr=0.0001_wd=0-epoch=039.ckpt',
            '1966': CHECKPOINT_DIR + r'final/modality_center/min_loss/09.03.2023-19.12HAM_SEED=1966_lr=0.0001_wd=0-epoch=039.ckpt',
            '3297': CHECKPOINT_DIR + r'final/modality_center/min_loss/09.03.2023-18.52HAM_SEED=3297_lr=0.0001_wd=0-epoch=039.ckpt', },
        'alpha_center': 0.01,
        'dropout': 0,
    },

    'cross_modal_center_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'final/new_center/min_loss/08.03.2023-04.19HAM_SEED=1997_lr=0.0001_wd=0-epoch=034.ckpt',
            '25': CHECKPOINT_DIR + r'final/new_center/min_loss/08.03.2023-04.37HAM_SEED=25_lr=0.0001_wd=0-epoch=034.ckpt',
            '12': CHECKPOINT_DIR + r'final/new_center/min_loss/08.03.2023-04.55HAM_SEED=12_lr=0.0001_wd=0-epoch=034.ckpt',
            '1966': CHECKPOINT_DIR + r'final/new_center/min_loss/08.03.2023-05.13HAM_SEED=1966_lr=0.0001_wd=0-epoch=034.ckpt',
            '3297': CHECKPOINT_DIR + r'final/new_center/min_loss/08.03.2023-05.31HAM_SEED=3297_lr=0.0001_wd=0-epoch=034.ckpt'},
        'alpha_center': 0.01,
        'dropout': 0,
    },

    'supervised_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'correlation': False,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'_SUPERVISED/train/27.05.2023-03.44_HAM_SEED=1997_lr=0.0001_wd=0-epoch=014.ckpt',
            '25': CHECKPOINT_DIR + r'_SUPERVISED/train/27.05.2023-03.59_HAM_SEED=25_lr=0.0001_wd=0-epoch=014.ckpt',
            '12': CHECKPOINT_DIR + r'_SUPERVISED/train/27.05.2023-04.15_HAM_SEED=12_lr=0.0001_wd=0-epoch=014.ckpt',
            '1966': CHECKPOINT_DIR + r'_SUPERVISED/train/27.05.2023-04.30_HAM_SEED=1966_lr=0.0001_wd=0-epoch=014.ckpt',
            '3297': CHECKPOINT_DIR + r'_SUPERVISED/train/27.05.2023-04.46_HAM_SEED=3297_lr=0.0001_wd=0-epoch=014.ckpt'},
        # resnet init
        # 'contrastive_checkpoint': {
        #     '25': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/gathered/18.04.2023-22.14_HAM_SEED=25_lr=0.0001_wd=0-epoch=099.ckpt',
        #     '1997': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/gathered/18.04.2023-21.08_HAM_SEED=1997_lr=0.0001_wd=0-epoch=099.ckpt',
        #     '12': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/gathered/18.04.2023-23.21_HAM_SEED=12_lr=0.0001_wd=0-epoch=099.ckpt',
        #     '1966': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/gathered/19.04.2023-00.27_HAM_SEED=1966_lr=0.0001_wd=0-epoch=099.ckpt',
        #     '3297': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/gathered/19.04.2023-01.34_HAM_SEED=3297_lr=0.0001_wd=0-epoch=099.ckpt'},
        # 'contrastive_checkpoint': {
        #     '25': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/random_init/gathered/20.04.2023-02.09_HAM_SEED=25_lr=0.0001_wd=0-epoch=099.ckpt',
        #     '1997': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/random_init/gathered/20.04.2023-01.01_HAM_SEED=1997_lr=0.0001_wd=0-epoch=099.ckpt',
        #     '12': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/random_init/gathered/20.04.2023-03.16_HAM_SEED=12_lr=0.0001_wd=0-epoch=099.ckpt',
        #     '1966': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/random_init/gathered/20.04.2023-04.22_HAM_SEED=1966_lr=0.0001_wd=0-epoch=099.ckpt',
        #     '3297': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/random_init/gathered/20.04.2023-05.27_HAM_SEED=3297_lr=0.0001_wd=0-epoch=099.ckpt'},
        # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised/min_lost/08.03.2023-04.15_HAM_SEED=1997_lr=0.0001_wd=0-epoch=011.ckpt',
        # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised/min_lost/08.03.2023-04.34_HAM_SEED=25_lr=0.0001_wd=0-epoch=011.ckpt',
        # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised/min_lost/08.03.2023-04.52_HAM_SEED=12_lr=0.0001_wd=0-epoch=011.ckpt',
        # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised/min_lost/08.03.2023-05.11_HAM_SEED=1966_lr=0.0001_wd=0-epoch=011.ckpt',
        # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised/min_lost/08.03.2023-05.30_HAM_SEED=3297_lr=0.0001_wd=0-epoch=011.ckpt',
        # 'checkpoint_corr': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised_corre/min_loss/08.03.2023-11.51_HAM_SEED=1997_lr=0.0001_wd=0-epoch=005.ckpt',
        # 'checkpoint_corr': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised_corre/min_loss/08.03.2023-12.09_HAM_SEED=25_lr=0.0001_wd=0-epoch=005.ckpt',
        # 'checkpoint_corr': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised_corre/min_loss/08.03.2023-12.26_HAM_SEED=12_lr=0.0001_wd=0-epoch=005.ckpt',
        # 'checkpoint_corr': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised_corre/min_loss/08.03.2023-12.44_HAM_SEED=1966_lr=0.0001_wd=0-epoch=005.ckpt',
        # 'checkpoint_corr': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised_corre/min_loss/08.03.2023-13.01_HAM_SEED=3297_lr=0.0001_wd=0-epoch=005.ckpt',
        # 'checkpoint_resnet': r'/home/guests/selen_erkan/experiments/checkpoints/final/resnet/min_lost/06.03.2023-19.08_FINAL_HAM_SEED=1997_lr=0.0001_wd=0-epoch=009.ckpt',
        # 'checkpoint_resnet': r'/home/guests/selen_erkan/experiments/checkpoints/final/resnet/min_lost/06.03.2023-19.40_FINAL_HAM_SEED=25_lr=0.0001_wd=0-epoch=009.ckpt',
        # 'checkpoint_resnet': r'/home/guests/selen_erkan/experiments/checkpoints/final/resnet/min_lost/06.03.2023-20.25_FINAL_HAM_SEED=12_lr=0.0001_wd=0-epoch=009.ckpt',
        # 'checkpoint_resnet': r'/home/guests/selen_erkan/experiments/checkpoints/final/resnet/min_lost/06.03.2023-21.12_FINAL_HAM_SEED=1966_lr=0.0001_wd=0-epoch=009.ckpt',
        # 'checkpoint_resnet': r'/home/guests/selen_erkan/experiments/checkpoints/final/resnet/min_lost/07.03.2023-16.23_FINAL_MISSING_HAM_SEED=3297_lr=0.0001_wd=0-epoch=009.ckpt',

    },
    'resnet_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,  # 40
        'age': None,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'_RESNET/train/27.05.2023-12.24_HAM_SEED=1997_lr=0.0001_wd=0-epoch=009.ckpt',
            '25': CHECKPOINT_DIR + r'_RESNET/train/27.05.2023-12.40_HAM_SEED=25_lr=0.0001_wd=0-epoch=009.ckpt',
            '12': CHECKPOINT_DIR + r'_RESNET/train/27.05.2023-12.55_HAM_SEED=12_lr=0.0001_wd=0-epoch=009.ckpt',
            '1966': CHECKPOINT_DIR + r'_RESNET/train/27.05.2023-13.10_HAM_SEED=1966_lr=0.0001_wd=0-epoch=009.ckpt',
            '3297': CHECKPOINT_DIR + r'_RESNET/train/27.05.2023-13.26_HAM_SEED=3297_lr=0.0001_wd=0-epoch=009.ckpt'}
    },

    'tabular_config': {
        'batch_size': 512,  # 512
        'max_epochs': 40,
        'age': None,
        'learning_rate': 1e-3,
        'weight_decay': 0,
        'checkpoint': {
            '1997': CHECKPOINT_DIR + r'_TABULAR/train/27.05.2023-19.05_HAM_SEED=1997_lr=0.001_wd=0-epoch=037.ckpt',
            '25': CHECKPOINT_DIR + r'_TABULAR/train/27.05.2023-19.15_HAM_SEED=25_lr=0.001_wd=0-epoch=037.ckpt',
            '12': CHECKPOINT_DIR + r'_TABULAR/train/27.05.2023-19.25_HAM_SEED=12_lr=0.001_wd=0-epoch=037.ckpt',
            '1966': CHECKPOINT_DIR + r'_TABULAR/train/27.05.2023-19.35_HAM_SEED=1966_lr=0.001_wd=0-epoch=037.ckpt',
            '3297': CHECKPOINT_DIR + r'_TABULAR/train/27.05.2023-19.45_HAM_SEED=3297_lr=0.001_wd=0-epoch=037.ckpt'}
    },


}
