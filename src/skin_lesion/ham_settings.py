import torch

root_dir = r'/home/guests/selen_erkan/datasets/skin_lesion'
image_dir = root_dir + r"/HAM10K_grouped_images"
csv_dir = root_dir + r"/HAM10000_metadata.csv"
CHECKPOINT_DIR = r'/home/guests/selen_erkan/experiments/checkpoints'

FEATURES = ['age', 'sex_numeric', 'label', 'abdomen', 'acral',	'back',	'chest', 'ear',	'face',	'foot',
            'genital',	'hand',	'lower extremity',	'neck',	'scalp',	'trunk',	'upper extremity']
# FEATURES = ['age', 'sex_numeric', 'label','localization_numeric']

TARGET = 'label'

SEED = 473

image_shape = (3, 224, 224)

supervised_config = {
    'batch_size': 681,  # 512
    'max_epochs': 100,
    'age': None,
    'learning_rate': 1e-5,
    'weight_decay': 0,
    'checkpoint': None,
    'contrastive_checkpoint': None,
    'checkpoint_flag': False,
    'contrastive_checkpoint_flag': False,
    'alpha_center': 0.01,
    'alpha_cross_ent': 0.99,
}

tabular_config = {
    'batch_size': 681,  # 512
    'max_epochs': 100,
    'age': None,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'checkpoint': None,
    'contrastive_checkpoint': None,
    'checkpoint_flag': False,
    'contrastive_checkpoint_flag': False
}
