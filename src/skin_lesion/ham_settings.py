import torch

root_dir = r'/home/guests/selen_erkan/datasets/skin_lesion'
image_dir = root_dir + r"/HAM10K_grouped_images"
csv_dir = root_dir + r"/HAM10000_metadata.csv"
train_dir = r'/train_data.csv'
test_dir = r'/test_data.csv'
CHECKPOINT_DIR = r'/home/guests/selen_erkan/experiments/checkpoints'

FEATURES = ['age', 'sex_numeric', 'label', 'abdomen', 'acral',	'back',	'chest', 'ear',	'face',	'foot',
            'genital',	'hand',	'lower extremity',	'neck',	'scalp',	'trunk',	'upper extremity']
# FEATURES = ['age', 'sex_numeric', 'label', 'localization_numeric']

TARGET = 'label'

seed_list = [1997, 25, 12, 1966, 3297]
SEED = 25

image_shape = (3, 224, 224)

triplet_center_config = {
    'batch_size': 512,  # 512
    'max_epochs': 40,  # 40
    'age': None,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'alpha_center': 0.01,
    'alpha_triplet': 0.3,
}

multiloss_config = {
    'batch_size': 512,  # 512
    'max_epochs': 40,  # 40
    'age': None,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss/lr_sch_23/07.03.2023-20.52HAM_PROPOSED_SEED=25_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss/lr_sch_23/07.03.2023-20.52HAM_PROPOSED_SEED=1997_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss/lr_sch_23/07.03.2023-20.53HAM_PROPOSED_SEED=12_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss/lr_sch_23/07.03.2023-21.14HAM_PROPOSED_SEED=1966_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss/lr_sch_23/07.03.2023-21.15HAM_PROPOSED_SEED=3297_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint_concat': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss_concat/min_loss/08.03.2023-04.07HAM_SEED=1997_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint_concat': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss_concat/min_loss/08.03.2023-04.25HAM_SEED=25_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint_concat': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss_concat/min_loss/08.03.2023-04.44HAM_SEED=12_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint_concat': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss_concat/min_loss/08.03.2023-05.03HAM_SEED=1966_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint_concat': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss_concat/min_loss/08.03.2023-05.22HAM_SEED=3297_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint_new_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/new_center/min_loss/08.03.2023-04.19HAM_SEED=1997_lr=0.0001_wd=0-epoch=034.ckpt',
    # 'checkpoint_new_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/new_center/min_loss/08.03.2023-04.37HAM_SEED=25_lr=0.0001_wd=0-epoch=034.ckpt',
    # 'checkpoint_new_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/new_center/min_loss/08.03.2023-04.55HAM_SEED=12_lr=0.0001_wd=0-epoch=034.ckpt',
    # 'checkpoint_new_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/new_center/min_loss/08.03.2023-05.13HAM_SEED=1966_lr=0.0001_wd=0-epoch=034.ckpt',
    # 'checkpoint_new_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/new_center/min_loss/08.03.2023-05.31HAM_SEED=3297_lr=0.0001_wd=0-epoch=034.ckpt',
    # 'checkpoint_modality_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/modality_center/min_loss/09.03.2023-18.51HAM_SEED=1997_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint_modality_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/modality_center/min_loss/09.03.2023-18.52HAM_SEED=12_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint_modality_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/modality_center/min_loss/09.03.2023-18.52HAM_SEED=3297_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint_modality_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/modality_center/min_loss/09.03.2023-19.10HAM_SEED=25_lr=0.0001_wd=0-epoch=039.ckpt',
    'checkpoint_modality_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/modality_center/min_loss/09.03.2023-19.12HAM_SEED=1966_lr=0.0001_wd=0-epoch=039.ckpt',
    'alpha_center': 0.01,
    'triplet_ratio': 0.7,
    # 'SEED': SEED,
    'dropout': 0,
}

supervised_config = {
    'batch_size': 512,  # 512
    'max_epochs': 40,  # 40
    'age': None,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'correlation': False,
    # resnet init
    # 'contrastive_checkpoint': {
    #     '25': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/gathered/18.04.2023-22.14_HAM_SEED=25_lr=0.0001_wd=0-epoch=099.ckpt',
    #     '1997': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/gathered/18.04.2023-21.08_HAM_SEED=1997_lr=0.0001_wd=0-epoch=099.ckpt',
    #     '12': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/gathered/18.04.2023-23.21_HAM_SEED=12_lr=0.0001_wd=0-epoch=099.ckpt',
    #     '1966': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/gathered/19.04.2023-00.27_HAM_SEED=1966_lr=0.0001_wd=0-epoch=099.ckpt',
    #     '3297': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/gathered/19.04.2023-01.34_HAM_SEED=3297_lr=0.0001_wd=0-epoch=099.ckpt'},
    # random init
    'contrastive_checkpoint': {
        '25': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/random_init/gathered/20.04.2023-02.09_HAM_SEED=25_lr=0.0001_wd=0-epoch=099.ckpt',
        '1997': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/random_init/gathered/20.04.2023-01.01_HAM_SEED=1997_lr=0.0001_wd=0-epoch=099.ckpt',
        '12': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/random_init/gathered/20.04.2023-03.16_HAM_SEED=12_lr=0.0001_wd=0-epoch=099.ckpt',
        '1966': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/random_init/gathered/20.04.2023-04.22_HAM_SEED=1966_lr=0.0001_wd=0-epoch=099.ckpt',
        '3297': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive_loss/seed/concat/random_init/gathered/20.04.2023-05.27_HAM_SEED=3297_lr=0.0001_wd=0-epoch=099.ckpt'},
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
    # 'checkpoint_daft': r'/home/guests/selen_erkan/experiments/checkpoints/final/daft/min_loss/08.03.2023-15.58HAM_SEED=1997_lr=0.0001_wd=0-epoch=007.ckpt',
    # 'checkpoint_daft': r'/home/guests/selen_erkan/experiments/checkpoints/final/daft/min_loss/08.03.2023-16.19HAM_SEED=25_lr=0.0001_wd=0-epoch=007.ckpt',
    # 'checkpoint_daft': r'/home/guests/selen_erkan/experiments/checkpoints/final/daft/min_loss/08.03.2023-16.40HAM_SEED=12_lr=0.0001_wd=0-epoch=007.ckpt',
    # 'checkpoint_daft': r'/home/guests/selen_erkan/experiments/checkpoints/final/daft/min_loss/08.03.2023-17.02HAM_SEED=1966_lr=0.0001_wd=0-epoch=007.ckpt',
    # 'checkpoint_daft': r'/home/guests/selen_erkan/experiments/checkpoints/final/daft/min_loss/08.03.2023-17.23HAM_SEED=3297_lr=0.0001_wd=0-epoch=007.ckpt',
    # 'checkpoint_film': r'/home/guests/selen_erkan/experiments/checkpoints/final/film/min_loss/09.03.2023-11.43HAM_SEED=1997_lr=0.0001_wd=0-epoch=006.ckpt',
    # 'checkpoint_film': r'/home/guests/selen_erkan/experiments/checkpoints/final/film/min_loss/09.03.2023-11.45HAM_SEED=12_lr=0.0001_wd=0-epoch=006.ckpt',
    # 'checkpoint_film': r'/home/guests/selen_erkan/experiments/checkpoints/final/film/min_loss/09.03.2023-11.47HAM_SEED=3297_lr=0.0001_wd=0-epoch=006.ckpt',
    # 'checkpoint_film': r'/home/guests/selen_erkan/experiments/checkpoints/final/film/min_loss/09.03.2023-12.08HAM_SEED=25_lr=0.0001_wd=0-epoch=006.ckpt',
    # 'checkpoint_film': r'/home/guests/selen_erkan/experiments/checkpoints/final/film/min_loss/09.03.2023-12.10HAM_SEED=1966_lr=0.0001_wd=0-epoch=006.ckpt',
    # 'SEED': SEED
}

tabular_config = {
    'batch_size': 512,  # 512
    'max_epochs': 40,
    'age': None,
    'learning_rate': 1e-3,
    'weight_decay': 0,
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/tabular/min_loss/08.03.2023-17.30_HAM_SEED=25_lr=0.001_wd=0-epoch=035.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/tabular/min_loss/08.03.2023-17.43_HAM_SEED=12_lr=0.001_wd=0-epoch=035.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/tabular/min_loss/08.03.2023-17.56_HAM_SEED=1966_lr=0.001_wd=0-epoch=035.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/tabular/min_loss/08.03.2023-18.02_HAM_SEED=1997_lr=0.001_wd=0-epoch=035.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/tabular/min_loss/08.03.2023-18.09_HAM_SEED=3297_lr=0.001_wd=0-epoch=035.ckpt',
    # # 'SEED': SEED
}

contrastive_loss_config = {

    'batch_size': 512,  # 512
    'max_epochs': 100,  # 40
    'age': None,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'correlation': False,
}
