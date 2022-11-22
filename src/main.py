from pytorch_lightning import Trainer

from model import AdniModel
from dataset import AdniDataModule

model = AdniModel()

module = AdniDataModule()

trainer = Trainer(max_epochs=1)

trainer.fit(model, module)
