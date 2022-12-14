import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from conv3D.model import AdniModel
from dataset import AdniDataModule
from multimodal_dataset import MultimodalDataModule

from ResNet.model import ResNetModel
from multimodal.model import MultiModModel


def main_conv3d(wandb, wandb_logger):
    '''
    main function to run the conv3d architecture
    '''
    # ge tthe model
    model = AdniModel()

    # load the data
    data = AdniDataModule()

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(max_epochs=15, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, data)


def main_resnet(wandb, wandb_logger):
    '''
    main function to run the resnet architecture
    '''
    # ge the model
    model = ResNetModel()

    # load the data
    data = AdniDataModule()

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(max_epochs=15, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, data)


def main_multimodal(wandb, wandb_logger):
    '''
    main function to run the multimodal architecture
    '''
    # ge the model
    model = MultiModModel()

    # load the data
    data = MultimodalDataModule()

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(max_epochs=20, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, data)


if __name__ == '__main__':

    # create wandb objects to track runs
    wandb.init(project="multimodal-network-test")
    wandb_logger = WandbLogger()

    # # run conv3d
    # main_conv3d(wandb, wandb_logger)

    # run resnet
    # main_resnet(wandb, wandb_logger)

    # run multimodal
    main_multimodal(wandb, wandb_logger)
