import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from conv3D.model import AdniModel
from dataset import AdniDataModule


def main_conv3d():
    '''
    main function to run the conv3d architecture
    '''

    wandb.init(project="multimodal-network-test")
    wandb.config = {
        "learning_rate": 1e-4,
        "epochs": 9,
        "batch_size": 1
    }

    # ge tthe model
    model = AdniModel()

    # load the data
    data = AdniDataModule()

    # Optional
    wandb.watch(model, log="all")

    # train the network
    wandb_logger = WandbLogger()
    trainer = Trainer(max_epochs=9, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, data)


if __name__ == '__main__':
    main_conv3d()
