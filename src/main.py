import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

from conv3D.model import AdniModel
from dataset import AdniDataModule
from multimodal_dataset import MultimodalDataModule, KfoldMultimodalDataModule
from contrastive_loss_dataset import ContrastiveDataModule

from models.resnet_model import ResNetModel
from models.multimodal_model import MultiModModel
from models.contrastive_learning_model import ContrastiveModel

import torch
from settings import CSV_FILE, SEED, CHECKPOINT_DIR, resnet_config, supervised_config, contrastive_config
import torch.multiprocessing
from datetime import datetime


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
    trainer = Trainer(max_epochs=15, logger=wandb_logger, deterministic=True)
    trainer.fit(model, data)


def main_resnet(config=None):
    '''
    main function to run the resnet architecture
    '''
    wandb.init(project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = ResNetModel(learning_rate=wandb.config.learning_rate,
                        weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(
        CSV_FILE, age=wandb.config.age, batch_size=wandb.config.batch_size, spatial_size=wandb.config.spatial_size)

    accelerator = 'cpu'
    devices = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1

    # save the checkpoint in a different folder
    # use datetime value in the file name
    date_time = datetime.now()
    dt_string = date_time.strftime("%d.%m.%Y-%H.%M")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'resnet'), filename=dt_string+'-{epoch:03d}')

    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10)
    trainer.fit(model, data)


def main_multimodal(config=None):
    '''
    main function to run the multimodal architecture
    '''
    wandb.init(project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = MultiModModel(learning_rate=wandb.config.learning_rate,
                          weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = MultimodalDataModule(
        CSV_FILE, age=wandb.config.age, batch_size=wandb.config.batch_size, spatial_size=wandb.config.spatial_size)

    accelerator = 'cpu'
    devices = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1

    # save the checkpoint in a different folder
    # use datetime value in the file name
    date_time = datetime.now()
    dt_string = date_time.strftime("%d.%m.%Y-%H.%M")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'supervised'), filename=dt_string+'-{epoch:03d}')

    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10)
    trainer.fit(model, data)


def main_kfold_multimodal(wandb, wandb_logger, fold_number=2, learning_rate=1e-3, batch_size=8, max_epochs=60, age=None):
    '''
    main function to run the multimodal architecture with cross validation
    '''

    # path to the csv file
    # this csv file contains image ids, patient ids and tabular info
    csv_dir = CSV_FILE

    # create kfold data object
    data_module = KfoldMultimodalDataModule(
        csv_dir, fold_number=fold_number, age=age, batch_size=batch_size)

    # get dataloaders for every fold
    train_dataloaders, val_dataloaders = data_module.prepare_data()

    train_fold_losses = []
    val_fold_losses = []
    accuracies = []

    accelerator = 'cpu'
    devices = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1

    fold_num = 0
    # train the mdoel
    for train_dataloader, val_dataloader in zip(train_dataloaders, val_dataloaders):
        # get the model
        model = MultiModModel(learning_rate=learning_rate)
        trainer = Trainer(accelerator=accelerator, devices=devices,
                          max_epochs=max_epochs, logger=wandb_logger, deterministic=True)
        trainer.fit(model, train_dataloader, val_dataloader)

        # log the loss of the fold
        wandb.log(
            {"train_fold_loss": model.metrics['train_epoch_losses'][-1], "fold": fold_num})
        wandb.log(
            {"val_fold_loss": model.metrics['val_epoch_losses'][-1], "fold": fold_num})

        # print the fold losses
        print(
            {'Fold {fold_num}, final train fold loss': model.metrics['train_epoch_losses'][-1]})
        print(
            {'Fold {fold_num}, final val fold loss': model.metrics['val_epoch_losses'][-1]})

        # add the final val and train losses to the list
        train_fold_losses.append(model.metrics['train_epoch_losses'][-1])
        val_fold_losses.append(model.metrics['val_epoch_losses'][-1])
        # accuracies.append(model.metrics["train_accuracy"][-1])

        # print(model.metrics['valid_accuracy'])
        # wandb.log({'Valid acc final': model.metrics['valid_accuracy'][-1]})
        # wandb.log({"Train acc final": model.metrics["train_accuracy"][-1]})

        fold_num += 1

    print('all the train losses: ', train_fold_losses)
    print('all the val losses: ', val_fold_losses)

    # log the average loss of folds
    wandb.log({"Average fold loss (val)": sum(
        val_fold_losses)/len(val_fold_losses)})
    wandb.log({"Average fold loss (train)":  sum(
        train_fold_losses)/len(train_fold_losses)})

    # wandb.log({"Valid acc avg": sum(
    #     model.metrics['valid_accuracy'])/len(model.metrics['valid_accuracy'])})
    # wandb.log({"Train acc avg":  sum(
    # model.metrics['train_accuracy'])/len(model.metrics['train_accuracy'])})
    # wandb.log({"Mean score":scores.mean()})


def main_contrastive_learning(config=None):
    '''
    main function to run the multimodal architecture
    '''
    wandb.init(project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = ContrastiveModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = ContrastiveDataModule(
        CSV_FILE, age=wandb.config.age, batch_size=wandb.config.batch_size, spatial_size=wandb.config.spatial_size)

    accelerator = 'cpu'
    devices = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1

    # save the checkpoint in a different folder
    # use datetime value in the file name
    date_time = datetime.now()
    dt_string = date_time.strftime("%d.%m.%Y-%H.%M")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'contrastive'), filename=dt_string+'-{epoch:03d}')

    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10)
    trainer.fit(model, data)


def run_grid_search(network):

    sweep_config = {
        'method': 'grid',
        'metric': {'goal': 'minimize', 'name': 'val_epoch_loss'},
        'parameters': {
            'network': {'value': 'supervised'},
            'batch_size': {'value': 16},
            'max_epochs': {'value': 10},
            'epochs': {'value': 5},
            'age': {'value': None},
            'spatial_size': {'value': (120, 120, 120)},
            'learning_rate': {'values': [0.01, 0.003, 0.001, 0.0003, 0.0001]},
            'weight_decay': {'values': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]},
        },
        'early_terminate': {'type': 'hyperband', 'min_iter': 3}
    }

    count = len(sweep_config.parameters.learning_rate) * \
        len(sweep_config.parameters.weight_decay)

    # sweep
    sweep_id = wandb.sweep(
        sweep_config, project="multimodal_training", entity="multimodal_network")
    wandb.agent(sweep_id, function=grid_search, count=count)
    wandb.finish()


def grid_search(config=None):
    '''
    main function to run grif search on the models
    '''

    with wandb.init(config=config):

        config = wandb.config
        wandb_logger = WandbLogger()
        wandb.watch(model, log="all")

        if config.network == 'resnet':
            # get the model
            model = ResNetModel(learning_rate=config.learning_rate,
                                weight_decay=config.weight_decay)
            # load the data
            data = AdniDataModule(
                CSV_FILE, age=config.age, batch_size=config.batch_size, spatial_size=config.spatial_size)

        elif config.network == 'supervised':
            # get the model
            model = MultiModModel(learning_rate=config.learning_rate,
                                  weight_decay=config.weight_decay)
            # load the data
            data = MultimodalDataModule(
                CSV_FILE, age=config.age, batch_size=config.batch_size, spatial_size=config.spatial_size)

        elif config.network == 'contrastive':
            # get the model
            model = MultiModModel(learning_rate=config.learning_rate,
                                  weight_decay=config.weight_decay)
            # load the data
            data = MultimodalDataModule(
                CSV_FILE, age=config.age, batch_size=config.batch_size, spatial_size=config.spatial_size)

        accelerator = 'cpu'
        devices = None
        if torch.cuda.is_available():
            accelerator = 'gpu'
            devices = 1

        trainer = Trainer(accelerator=accelerator, devices=devices,
                          max_epochs=config.max_epochs, logger=wandb_logger, log_every_n_steps=10)
        trainer.fit(model, data)


if __name__ == '__main__':

    # set the seed of the environment
    # Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    seed_everything(SEED, workers=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # run resnet
    # main_resnet(resnet_config)

    # run multimodal
    main_multimodal(supervised_config)

    # run contrastive learning
    # main_contrastive_learning(contrastive_config)

    # run kfold multimodal
    # main_kfold_multimodal(wandb, wandb_logger, fold_number = 5, learning_rate=1e-3, batch_size=8, max_epochs=100, age=None)

    # run grid search
    # run_grid_search('supervised')
