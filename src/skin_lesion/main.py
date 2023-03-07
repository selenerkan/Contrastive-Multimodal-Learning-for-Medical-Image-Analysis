import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from ham_settings import csv_dir, supervised_config, CHECKPOINT_DIR, SEED, tabular_config, multiloss_config
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
import random
from models.ham_daft_model import DaftModel
from models.ham_new_center_model import NewCenterModel


def main_baseline(config=None):
    '''
    main function to run the baseline model
    '''

    print('YOU ARE RUNNING BASELINE FOR HAM DATASET')
    print(config)

    wandb.init(group='HAM_baseline',
               project="final_multimodal_training", config=config)
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
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[lr_monitor], deterministic=True)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def main_resnet(config=None):
    '''
    main function to run the baseline model
    '''
    print('YOU ARE RUNNING RESNET FOR HAM DATASET')
    print(config)

    wandb.init(group='SEED_MISSING_HAM_resnet',
               project="final_multimodal_training", config=config)
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

    # save the checkpoint in a different folder
    # use datetime value in the file name
    date_time = datetime.now()
    dt_string = date_time.strftime("%d.%m.%Y-%H.%M")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'resnet'),
        filename=dt_string+'_FINAL_MISSING_HAM_SEED='+str(SEED)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
        str(wandb.config.weight_decay)+'-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=True)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def test_resnet(config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR RESNET MODEL FOR HAM DATASET')

    wandb.init(group='TEST_MIN_LOSS_HAM_resnet',
               project="final_multimodal_training", config=config)
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
    test_dataloader = data.test_dataloader()

    accelerator = 'cpu'
    devices = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[lr_monitor], deterministic=True)
    trainer.test(model, dataloaders=test_dataloader,
                 ckpt_path=wandb.config.checkpoint_resnet)


def main_tabular(config=None):
    '''
    main function to run the baseline model
    '''

    print('YOU ARE RUNNING TABULAR MODEL FOR HAM DATASET')
    print(config)

    wandb.init(group='HAM_tabular',
               project="final_multimodal_training", config=config)
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
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[lr_monitor], deterministic=True)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def main_supervised_multimodal(config=None):
    '''
    main function to run the supervised multimodal architecture
    '''

    print('YOU ARE RUNNING SEED SUPERVISED MULTIMODAL FOR HAM DATASET')
    print(config)

    wandb.init(group='SEED_CORR_HAM_supervised',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = SupervisedModel(learning_rate=wandb.config.learning_rate,
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

    # save the checkpoint in a different folder
    # use datetime value in the file name
    date_time = datetime.now()
    dt_string = date_time.strftime("%d.%m.%Y-%H.%M")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'supervised'),
        filename=dt_string+'_CORRELATION_HAM_SEED='+str(SEED)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
        str(wandb.config.weight_decay)+'-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=20,
        mode='max')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=True)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def test_supervised_multimodal(config=None):
    '''
    main function to run the test loop for supervised multimodal architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FORSUPERVISED MULTIMODAL FOR HAM DATASET')

    wandb.init(group='TEST_fINAL_HAM_supervised',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = SupervisedModel(learning_rate=wandb.config.learning_rate,
                            weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data()
    data.set_supervised_multimodal_dataloader()
    test_dataloader = data.test_dataloader()

    accelerator = 'cpu'
    devices = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[lr_monitor], deterministic=True)
    trainer.test(model, dataloaders=test_dataloader,
                 ckpt_path=wandb.config.checkpoint)


def main_multiloss(config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING SEED MULTI LOSS MODEL CENTER + CROSS ENTROPY LOSSES FOR HAM DATASET')
    print(config)

    wandb.init(group='SEED_LR_SCHEDULER_LONGER_PROPOSED_METHOD_HAM',
               project="final_multimodal_training",  config=config)
    wandb_logger = WandbLogger()

    model = MultiLossModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout)
    # use this only for center loss with dim=0 concat
    # model = NewCenterModel(learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout)
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

    # save the checkpoint in a different folder
    # use datetime value in the file name
    date_time = datetime.now()
    dt_string = date_time.strftime("%d.%m.%Y-%H.%M")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'multi_loss/lr_scheduler_longer'),
        filename=dt_string+'HAM_PROPOSED_SEED='+str(SEED)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
        str(wandb.config.weight_decay)+'-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=True)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def test_multiloss(config=None):
    '''
    main function to run the test loop for multiloss architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR MULTILOSS FOR HAM DATASET')

    wandb.init(group='TEST_LR_23_PROPOSED_HAM',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    # CONCAT
    model = MultiLossModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout)
    # use this only for center loss with dim=0 concat
    # model = NewCenterModel(learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout)

    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data()
    data.set_supervised_multimodal_dataloader()
    test_dataloader = data.test_dataloader()

    accelerator = 'cpu'
    devices = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[lr_monitor], deterministic=True)
    trainer.test(model, dataloaders=test_dataloader,
                 ckpt_path=wandb.config.checkpoint)
    # trainer.test(model, dataloaders=test_dataloader,
    #              ckpt_path=wandb.config.checkpoint_concat)


def run_grid_search(network):

    print('YOU ARE RUNNING GRID SEARCH FOR: ', network)

    sweep_config = {
        'method': 'grid',
        'metric': {'goal': 'maximize', 'name': 'val_macro_acc'},
        'parameters': {
            'network': {'value': network},
            'batch_size': {'value': 512},
            'max_epochs': {'value': 60},
            'age': {'value': None},
            'learning_rate': {'values': [1e-4, 1e-3]},
            'weight_decay': {'value': 0},
            'alpha_center': {'values': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]},
            'dropout': {'value': 0},
            'triplet_ratio': {'value': 0},
        }
    }

    # sweep_config = {
    #     'method': 'grid',
    #     'metric': {'goal': 'maximize', 'name': 'val_macro_acc'},
    #     'parameters': {
    #         'network': {'value': network},
    #         'batch_size': {'value': 512},
    #         'max_epochs': {'value': 30},
    #         'age': {'value': None},
    #         'learning_rate': {'values': [1e-4]},
    #         'weight_decay': {'value': 0},
    #         'dropout': {'value': 0},
    #         'alpha_center': {'values': [0.01, 0.05, 0.1, 0.2]},
    #         'triplet_ratio': {'values': [0.05, 0.1, 0.2, 0.5]}
    #     }
    # }

    # sweep_config = {
    #     'method': 'grid',
    #     'metric': {'goal': 'maximize', 'name': 'val_macro_acc'},
    #     'parameters': {
    #         'network': {'value': network},
    #         'batch_size': {'value': 512},
    #         'max_epochs': {'value': 30},
    #         'age': {'value': None},
    #         'learning_rate': {'values': [1e-4]},
    #         'weight_decay': {'value': 0},
    #         'dropout': {'value': 0},
    #         'alpha_center': {'values': [0.01, 0.05, 0.1, 0.2]},
    #         'alpha_adv': {'values': [0.1, 0.2, 0.5, 0.7]}
    #     }
    # }

    # sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project="final_multimodal_training", )
    wandb.agent(sweep_id, function=main_multiloss)
    wandb.finish()


def main_daft(config=None):
    '''
    main function to run the supervised multimodal DAFT architecture
    '''

    print('YOU ARE RUNNING DAFT FOR HAM DATASET')
    print(config)

    wandb.init(group='HAM_daft_3_Tabular',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = DaftModel(learning_rate=wandb.config.learning_rate,
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

    # save the checkpoint in a different folder
    # use datetime value in the file name
    date_time = datetime.now()
    dt_string = date_time.strftime("%d.%m.%Y-%H.%M")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'daft'),
        filename=dt_string+'HAM_DAFT_3_TABULAR_SEED='+str(SEED)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
        str(wandb.config.weight_decay)+'-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=True)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


if __name__ == '__main__':

    # set the seed of the environment
    # Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    seed_everything(SEED, workers=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    # run baseline
    # main_baseline(supervised_config)

    # run resnet baseline
    # main_resnet(supervised_config)

    # run tabular baseline
    # main_tabular(tabular_config)

    # run multimodal
    # main_supervised_multimodal(supervised_config)

    # run multiloss model (center + cross entropy + triplet)
    main_multiloss(multiloss_config)

    # run grid search
    # run_grid_search('multi_loss')

    # run daft
    # main_daft(supervised_config)

    # TESTING
    # test_supervised_multimodal(supervised_config)
    # test_resnet(supervised_config)
    # test_multiloss(multiloss_config)
