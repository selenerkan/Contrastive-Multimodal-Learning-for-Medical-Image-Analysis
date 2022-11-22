# from model import LitMNIST
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

import torchvision.transforms as transforms
from dataset import Adni_Dataset
from torch.utils.data import DataLoader

from settings import CSV_FILE, IMAGE_PATH, IMAGE_SIZE
from pytorch_lightning import Trainer

from model import AdniModel
from dataset import AdniDataModule

model = AdniModel()

module = AdniDataModule()

# Don't go over 10000 - 100000 or it will take 5 - 53+ hours to iterate
trainer = Trainer(max_epochs=1)

trainer.fit(model, module)

# training_transformations = transforms.Compose([
#     # transforms.ToPILImage(),
#     transforms.ToTensor()])

# target_transformations = None

# train = Adni_Dataset(CSV_FILE + '\labels.csv', IMAGE_PATH,
#                      training_transformations, target_transformations)

# dataloader = DataLoader(train, batch_size=1, shuffle=True)

# inputs, classes = next(iter(dataloader))

# print(inputs, classes)
