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

trainer = Trainer(max_epochs=1)

trainer.fit(model, module)
