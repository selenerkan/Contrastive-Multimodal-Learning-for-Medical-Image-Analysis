import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from ham_settings import csv_dir, supervised_config, CHECKPOINT_DIR, SEED, tabular_config
from models.ham_supervised_model import SupervisedModel
from models.image_model import BaselineModel
from models.resnet_model import ResnetModel
from models.tabular_model import TabularModel
from models.ham_multi_loss_model import MultiLossModel
from ham_dataset import HAMDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import torch.multiprocessing
from datetime import datetime
import random
import numpy as np


def main_baseline(config=None):
    '''
    main function to run the baseline model
    '''

    print('YOU ARE RUNNING BASELINE FOR HAM DATASET')
    print(config)

    wandb.init(group='HAM_baseline', project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = BaselineModel(learning_rate=wandb.config.learning_rate,
                          weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data()
    data.set_supervised_multimodal_dataloader()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    accelerator = 'cpu'
    devices = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[lr_monitor], log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def main_resnet(config=None):
    '''
    main function to run the baseline model
    '''

    print('YOU ARE RUNNING RESNET FOR HAM DATASET')
    print(config)

    wandb.init(group='HAM_resnet', project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = ResnetModel(learning_rate=wandb.config.learning_rate,
                        weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data()
    data.set_supervised_multimodal_dataloader()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    accelerator = 'cpu'
    devices = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[lr_monitor], log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def main_tabular(config=None):
    '''
    main function to run the baseline model
    '''

    print('YOU ARE RUNNING TABULAR MODEL FOR HAM DATASET')
    print(config)

    wandb.init(group='HAM_tabular', project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = TabularModel(learning_rate=wandb.config.learning_rate,
                         weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data()
    data.set_supervised_multimodal_dataloader()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    accelerator = 'cpu'
    devices = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[lr_monitor], log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def main_supervised_multimodal(config=None):
    '''
    main function to run the supervised multimodal architecture
    '''

    print('YOU ARE RUNNING SUPERVISED MULTIMODAL FOR HAM DATASET')
    print(config)

    wandb.init(group='HAM_supervised', project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = SupervisedModel(learning_rate=wandb.config.learning_rate,
                            weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # check if the checkpoint flag is True
    if wandb.config.checkpoint_flag:
        print('YOU ARE USING A CHECKPOINT OF A SUPERVISED MULTIMODAL NETWORK')
        # copy the weights from multimodal supervised model checkpoint
        model = SupervisedModel.load_from_checkpoint(
            config.checkpoint, learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    # elif wandb.config.contrastive_checkpoint_flag:
    #     print('YOU ARE USING A CONTRASTIVE LEARNING WEIGHTS ON SUPERVISED MULTIMODAL NETWORK FOR FINETUNING')
    #     contrastive_model = ContrastiveModel.load_from_checkpoint(
    #         wandb.config.contrastive_checkpoint)
    #     # copy the resnet and fc1 weights from contrastive learning model (pretrainening)
    #     model.resnet = contrastive_model.resnet
    #     model.fc1 = contrastive_model.fc1

        # freeze network weights (uncomment if you want to freeze the network weights)
        # model.resnet.freeze()
        # model.fc1.requires_grad_(False)

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data()
    data.set_supervised_multimodal_dataloader()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

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
        dirpath=os.path.join(CHECKPOINT_DIR, 'supervised'), filename='HAM_lr='+str(wandb.config.learning_rate)+'_wd='+str(wandb.config.weight_decay)+'_'+dt_string+'-{epoch:03d}')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def main_multiloss(config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING MULTI LOSS MODEL WITH CENTER + CROSS ENTROPY LOSSES FOR HAM DATASET')
    print(config)

    wandb.init(group='HAM_center_cross_ent', project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the modela
    model = MultiLossModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, alpha_cross_ent=wandb.config.alpha_cross_ent)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data()
    data.set_triplet_dataloader()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

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
        dirpath=os.path.join(CHECKPOINT_DIR, 'multi_loss'), filename='HAM_lr='+str(wandb.config.learning_rate)+'_wd='+str(wandb.config.weight_decay)+'_'+dt_string+'-{epoch:03d}')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def run_grid_search(network):

    print('YOU ARE RUNNING GRID SEARCH FOR: ', network)

    sweep_config = {
        'method': 'grid',
        'metric': {'goal': 'minimize', 'name': 'val_epoch_loss'},
        'parameters': {
            'network': {'value': network},
            'batch_size': {'value': 681},
            'max_epochs': {'value': 50},
            'age': {'value': None},
            'learning_rate': {'values': [1e-4, 1e-5]},
            'weight_decay': {'value': 0},
            'alpha_center': {'values': [0.001, 0.01, 0.05, 0.1, 0.2]},
        }
    }

    # sweep_config = {
    #     'method': 'grid',
    #     'metric': {'goal': 'minimize', 'name': 'val_epoch_loss'},
    #     'parameters': {
    #         'network': {'value': network},
    #         'batch_size': {'value': 681},
    #         'max_epochs': {'value': 50},
    #         'age': {'value': None},
    #         'learning_rate': {'values': [1e-3,1e-4,1e-5]},
    #         'weight_decay': {'values': [0, 1e-4,1e-3,1e-2]},
    #     }
    # }

    count = len(sweep_config['parameters']['learning_rate']['values']) * \
        len(sweep_config['parameters']['alpha_center']['values'])
    # count = len(sweep_config['parameters']['learning_rate']['values']) *len(sweep_config['parameters']['weight_decay']['values'])

    # sweep
    # sweep_id = wandb.sweep(
    #     sweep_config, project="multimodal_training", entity="multimodal_network")
    sweep_id = "zvtumpa3"
    wandb.agent(sweep_id, project="multimodal_training",
                entity="multimodal_network", function=grid_search)
    wandb.finish()


def grid_search(config=None):
    '''
    main function to run grid search on the models
    '''
    with wandb.init(config=config):

        config = wandb.config
        wandb_logger = WandbLogger()

        # load the data
        data = HAMDataModule(
            csv_dir, age=config.age, batch_size=config.batch_size)
        data.prepare_data()

        if config.network == 'resnet':
            # get the model
            model = ResnetModel(learning_rate=config.learning_rate,
                                weight_decay=config.weight_decay)
            data.set_supervised_multimodal_dataloader()

        elif config.network == 'tabular':
            # get the model
            model = TabularModel(learning_rate=config.learning_rate,
                                 weight_decay=config.weight_decay)
            data.set_supervised_multimodal_dataloader()

        elif config.network == 'supervised':
            # get the model
            model = SupervisedModel(learning_rate=config.learning_rate,
                                    weight_decay=config.weight_decay)
            data.set_supervised_multimodal_dataloader()

        elif config.network == 'multi_loss':
            # get the model
            model = MultiLossModel(
                learning_rate=config.learning_rate, weight_decay=config.weight_decay, alpha_center=config.alpha_center)
            # load the data
            data.set_triplet_dataloader()

        # get dataloaders
        train_dataloader = data.train_dataloader()
        val_dataloader = data.val_dataloader()

        wandb.watch(model, log="all")
        accelerator = 'cpu'
        devices = None
        if torch.cuda.is_available():
            accelerator = 'gpu'
            devices = 1

        # save the checkpoint in a model specific folder
        # use datetime value in the file name
        date_time = datetime.now()
        dt_string = date_time.strftime("%d.%m.%Y-%H.%M")
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(CHECKPOINT_DIR, config.network), filename='grid_lr='+str(config.learning_rate)+'_wd='+str(config.weight_decay)+'_'+dt_string+'-{epoch:03d}')

        # Add learning rate scheduler monitoring
        trainer = Trainer(accelerator=accelerator, devices=devices,
                          max_epochs=config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10)
        # trainer.fit(model, data)
        trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    # set the seed of the environment
    # Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    seed_everything(SEED, workers=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # run baseline
    # main_baseline(supervised_config)

    # run resnet baseline
    # main_resnet(supervised_config)

    # run tabular baseline
    # main_tabular(tabular_config)

    # run multimodal
    # main_supervised_multimodal(supervised_config)

    # run multiloss model (center + cross entropy + triplet)
    # main_multiloss(supervised_config)

    # run grid search
    run_grid_search('multi_loss')
