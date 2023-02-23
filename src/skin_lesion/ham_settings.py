import torch 

root_dir = r'/home/guests/selen_erkan/datasets/skin_lesion'
image_dir = root_dir + r"/HAM10K_grouped_images"
csv_dir = root_dir + r"/HAM10000_metadata.csv"
CHECKPOINT_DIR = r'/home/guests/selen_erkan/experiments/checkpoints'

# class_weights = {'akiec': 1.5565749235474007,
#                  'bcc': 1.0,
#                  'bkl': 0.47304832713754646,
#                  'df': 4.426086956521739,
#                  'mel': 0.4614687216681777,
#                  'nv': 0.0783197414986921,
#                  'vasc': 3.584507042253521}

class_weights = torch.tensor([1.5565749235474007,
                              1.0,
                              0.47304832713754646,
                              4.426086956521739,
                              0.4614687216681777,
                              0.0783197414986921,
                              3.584507042253521])

FEATURES = ['age', 'sex_numeric', 'localization_numeric', 'label']

TARGET = 'label'

SEED = 473

image_shape = (3, 224, 224)

supervised_config = {
    'batch_size': 32,
    'max_epochs': 40,
    'age': None,
    'learning_rate': 1e-5,
    'weight_decay': 0,
    'checkpoint': None,
    # 'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive/lr=0.001_wd=0_27.01.2023-17.49-epoch=079.ckpt',
    'contrastive_checkpoint': r'/home/guests/selen_erkan/experiments/checkpoints/triplet/lr=0.013_wd=0.01_01.02.2023-17.19-epoch=020.ckpt',
    'checkpoint_flag': False,
    'contrastive_checkpoint_flag': False
}
