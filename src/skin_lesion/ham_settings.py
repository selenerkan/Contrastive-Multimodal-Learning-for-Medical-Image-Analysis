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
SEED = 3297

image_shape = (3, 224, 224)

multiloss_config = {
    'batch_size': 512,  # 512
    'max_epochs': 70,  # 40
    'age': None,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss/lr_sch_23/07.03.2023-20.52HAM_PROPOSED_SEED=25_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss/lr_sch_23/07.03.2023-20.52HAM_PROPOSED_SEED=1997_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss/lr_sch_23/07.03.2023-20.53HAM_PROPOSED_SEED=12_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss/lr_sch_23/07.03.2023-21.14HAM_PROPOSED_SEED=1966_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss/lr_sch_23/07.03.2023-21.15HAM_PROPOSED_SEED=3297_lr=0.0001_wd=0-epoch=039.ckpt',
    # 'checkpoint_concat': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss_concat/06.03.2023-22.32_FINAL_HAM_CONCAT_CC_SEED=1997_lr=0.0001_wd=0-epoch=020.ckpt',
    # 'checkpoint_concat': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss_concat/06.03.2023-22.51_FINAL_HAM_CONCAT_CC_SEED=1966_lr=0.0001_wd=0-epoch=020.ckpt',
    # 'checkpoint_concat': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss_concat/06.03.2023-22.52_FINAL_HAM_CONCAT_CC_SEED=3297_lr=0.0001_wd=0-epoch=020.ckpt',
    # 'checkpoint_concat': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss_concat/07.03.2023-00.31_FINAL_MISSING_CONCAT_HAM_CC_SEED=25_lr=0.0001_wd=0-epoch=020.ckpt',
    # 'checkpoint_concat': r'/home/guests/selen_erkan/experiments/checkpoints/final/multi_loss_concat/07.03.2023-00.32_FINAL_MISSING_CONCAT_HAM_CC_SEED=12_lr=0.0001_wd=0-epoch=020.ckpt',
    # 'checkpoint_new_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/new_center/18/06.03.2023-23.12_FINAL_HAM_NEW_CENTER_CONCAT_SEED=25_lr=0.0001_wd=0-epoch=018.ckpt',
    # 'checkpoint_new_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/new_center/18/06.03.2023-23.13_FINAL_HAM_NEW_CENTER_CONCAT_SEED=12_lr=0.0001_wd=0-epoch=018.ckpt',
    # 'checkpoint_new_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/new_center/18/06.03.2023-23.33_FINAL_HAM_NEW_CENTER_CONCAT_SEED=1966_lr=0.0001_wd=0-epoch=018.ckpt',
    # 'checkpoint_new_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/new_center/18/06.03.2023-23.34_FINAL_HAM_NEW_CENTER_CONCAT_SEED=3297_lr=0.0001_wd=0-epoch=018.ckpt',
    # 'checkpoint_new_center': r'/home/guests/selen_erkan/experiments/checkpoints/final/new_center/18/07.03.2023-15.06MISSING_HAM_NEW_CENTER_CC_SEED=1997_lr=0.0001_wd=0-epoch=018.ckpt',
    'alpha_center': 0.01,
    'triplet_ratio': 0.7,
    'SEED': SEED,
    'dropout': 0,
}

supervised_config = {
    'batch_size': 512,  # 512
    'max_epochs': 40,  # 40
    'age': None,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised/06.03.2023-19.11_FINAL_HAM_SEED=1997_lr=0.0001_wd=0-epoch=020.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised/06.03.2023-19.41_FINAL_HAM_SEED=25_lr=0.0001_wd=0-epoch=020.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised/06.03.2023-20.26_FINAL_HAM_SEED=12_lr=0.0001_wd=0-epoch=020.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised/06.03.2023-21.11_FINAL_HAM_SEED=1966_lr=0.0001_wd=0-epoch=020.ckpt',
    # 'checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/final/supervised/06.03.2023-23.52_MISSING_HAM_SEED=3297_lr=0.0001_wd=0-epoch=020.ckpt',
    # 'checkpoint_resnet': r'/home/guests/selen_erkan/experiments/checkpoints/final/resnet/min_lost/06.03.2023-19.08_FINAL_HAM_SEED=1997_lr=0.0001_wd=0-epoch=009.ckpt',
    # 'checkpoint_resnet': r'/home/guests/selen_erkan/experiments/checkpoints/final/resnet/min_lost/06.03.2023-19.40_FINAL_HAM_SEED=25_lr=0.0001_wd=0-epoch=009.ckpt',
    # 'checkpoint_resnet': r'/home/guests/selen_erkan/experiments/checkpoints/final/resnet/min_lost/06.03.2023-20.25_FINAL_HAM_SEED=12_lr=0.0001_wd=0-epoch=009.ckpt',
    # 'checkpoint_resnet': r'/home/guests/selen_erkan/experiments/checkpoints/final/resnet/min_lost/06.03.2023-21.12_FINAL_HAM_SEED=1966_lr=0.0001_wd=0-epoch=009.ckpt',
    # 'checkpoint_resnet': r'/home/guests/selen_erkan/experiments/checkpoints/final/resnet/min_lost/07.03.2023-16.23_FINAL_MISSING_HAM_SEED=3297_lr=0.0001_wd=0-epoch=009.ckpt',
    'SEED': SEED
}

tabular_config = {
    'batch_size': 512,  # 512
    'max_epochs': 100,
    'age': None,
    'learning_rate': 1e-3,
    'weight_decay': 0,
    'checkpoint': None,
    'contrastive_checkpoint': None,
    'checkpoint_flag': False,
    'contrastive_checkpoint_flag': False,
    'SEED': SEED
}
