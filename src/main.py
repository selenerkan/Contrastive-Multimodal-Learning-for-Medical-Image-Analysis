from pytorch_lightning import Trainer

from model import AdniModel
from dataset import AdniDataModule

# ge tthe model
model = AdniModel()

# load the data
data = AdniDataModule()

# train the network
trainer = Trainer(max_epochs=1)
trainer.fit(model, data)
