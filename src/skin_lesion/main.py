import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from ham_settings import csv_dir, supervised_config, CHECKPOINT_DIR, tabular_config, multiloss_config, seed_list, contrastive_loss_config, triplet_center_config
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
from models.ham_film_model import FilmModel
from models.ham_new_center_model import NewCenterModel
from models.ham_modality_specific_center import ModalityCenterModel
from models.ham_contrastive_loss_model import HamContrastiveModel
from models.ham_triplet_center_cross_ent import TripletCenterModel


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


def main_resnet(seed, config=None):
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
        filename=dt_string+'_FINAL_MISSING_HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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


def test_resnet(seed, config=None):
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


def main_tabular(seed, config=None):
    '''
    main function to run the baseline model
    '''

    print('YOU ARE RUNNING TABULAR MODEL FOR HAM DATASET')
    print(config)

    wandb.init(group='FULL_NEWWW_HAM_tabular',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = TabularModel(learning_rate=wandb.config.learning_rate,
                         weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed)
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'tabular/full'),
        filename=dt_string+'_HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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
    wandb.finish()


def main_supervised_multimodal(seed, config=None):
    '''
    main function to run the supervised multimodal architecture
    '''

    print('YOU ARE RUNNING SEED SUPERVISED MULTIMODAL FOR HAM DATASET')
    print(config)

    wandb.init(group='SEED_SUPERVISED_RANDOM_INIT_CONCAT',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = SupervisedModel(learning_rate=wandb.config.learning_rate,
                            weight_decay=wandb.config.weight_decay, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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
        dirpath=os.path.join(
            CHECKPOINT_DIR, 'supervised/random_init/concat/seed/'),
        filename=dt_string+'_HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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

    wandb.finish()


def test_supervised_multimodal(seed, config=None):
    '''
    main function to run the test loop for supervised multimodal architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FORSUPERVISED MULTIMODAL FOR HAM DATASET')

    wandb.init(group='TEST_FULL_NEWWW_SUPERVISED_CONCAT_HAM',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = SupervisedModel(learning_rate=wandb.config.learning_rate,
                            weight_decay=wandb.config.weight_decay, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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


def test_supervised_corr_multimodal(seed, config=None):
    '''
    main function to run the test loop for supervised multimodal architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FORSUPERVISED MULTIMODAL FOR HAM DATASET')

    wandb.init(group='TEST_FULL_NEWW_SUPERVISED_CORR_HAM',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = SupervisedModel(learning_rate=wandb.config.learning_rate,
                            weight_decay=wandb.config.weight_decay, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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
                 ckpt_path=wandb.config.checkpoint_corr)


def main_multiloss(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING SEED MULTI LOSS MODEL CENTER + CROSS ENTROPY LOSSES FOR HAM DATASET')
    print(config)

    wandb.init(group='SEED_FULL_CONCAT_HAM',
               project="final_multimodal_training",  config=config)
    wandb_logger = WandbLogger()

    model = MultiLossModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout, seed=seed)
    # use this only for center loss with dim=0 concat
    # model = NewCenterModel(learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'multi_loss/concat/full'),
        filename=dt_string+'HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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
    wandb.finish()


def test_multiloss(seed, config=None):
    '''
    main function to run the test loop for multiloss architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR MULTILOSS FOR HAM DATASET')

    wandb.init(group='TEST_FULL_CONCAT_HAM',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    # CONCAT
    model = MultiLossModel(seed=seed,
                           learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout)
    # use this only for center loss with dim=0 concat
    # model = NewCenterModel(learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout)

    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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
    # trainer.test(model, dataloaders=test_dataloader,
    #              ckpt_path=wandb.config.checkpoint)
    trainer.test(model, dataloaders=test_dataloader,
                 ckpt_path=wandb.config.checkpoint_concat)


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


def main_daft(seed, config=None):
    '''
    main function to run the supervised multimodal DAFT architecture
    '''

    print('YOU ARE RUNNING DAFT FOR HAM DATASET')
    print(config)

    wandb.init(group='FULL_ONLY_DAFT',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = DaftModel(learning_rate=wandb.config.learning_rate,
                      weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed)
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'daft/full/only_daft'),
        filename=dt_string+'HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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
    wandb.finish()


def test_daft(seed, config):
    print('YOU ARE RUNNING DAFT FOR HAM DATASET')
    print(config)

    wandb.init(group='TEST_NEW_DAFT',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = DaftModel(learning_rate=wandb.config.learning_rate,
                      weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed)
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
                 ckpt_path=wandb.config.checkpoint_daft)
    wandb.finish()


def main_new_center(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING SEED MULTI LOSS MODEL CENTER + CROSS ENTROPY LOSSES FOR HAM DATASET')
    print(config)

    wandb.init(group='SEED_FULL_NEW_CENTER_HAM',
               project="final_multimodal_training",  config=config)
    wandb_logger = WandbLogger()

    model = NewCenterModel(learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay,
                           alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout, seed=seed)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'multi_loss/new_center/full'),
        filename=dt_string+'HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
        str(wandb.config.weight_decay)+'-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=True)
    # trainer.test(model, dataloaders=test_dataloader,
    #              ckpt_path=wandb.config.checkpoint_new_center)
    wandb.finish()


def test_new_center(seed, config=None):
    '''
    main function to run the test loop for multiloss architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR MULTILOSS FOR HAM DATASET')

    wandb.init(group='TEST_FULL_NEWWW_NEW_CENTER_HAM',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    # CONCAT
    model = NewCenterModel(seed=seed,
                           learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout)
    # use this only for center loss with dim=0 concat
    # model = NewCenterModel(learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout)

    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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
    # trainer.test(model, dataloaders=test_dataloader,
    #              ckpt_path=wandb.config.checkpoint)
    trainer.test(model, dataloaders=test_dataloader,
                 ckpt_path=wandb.config.checkpoint_new_center)


def test_tabular(seed, config):

    print('YOU ARE RUNNING TABULAR MODEL FOR HAM DATASET')
    print(config)

    wandb.init(group='TEST_HAM_tabular',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = TabularModel(learning_rate=wandb.config.learning_rate,
                         weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed)
    data.set_supervised_multimodal_dataloader()
    test_dataloader = data.test_dataloader()
    # val_dataloader = data.val_dataloader()

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


def main_film(seed, config):
    '''
    main function to run the supervised multimodal DAFT architecture
    '''

    print('YOU ARE RUNNING FILM FOR HAM DATASET')
    print(config)

    wandb.init(group='FULL_FILM',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = FilmModel(learning_rate=wandb.config.learning_rate,
                      weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed)
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'film/full'),
        filename=dt_string+'HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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
    wandb.finish()


def test_film(seed, config):
    print('YOU ARE RUNNING FILM FOR HAM DATASET')
    print(config)

    wandb.init(group='TEST_FILM',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = FilmModel(learning_rate=wandb.config.learning_rate,
                      weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed)
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
                 ckpt_path=wandb.config.checkpoint_film)
    wandb.finish()


def main_modality_center(seed, config):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING SEED MULTI LOSS MODEL CENTER + CROSS ENTROPY LOSSES FOR HAM DATASET')
    print(config)

    wandb.init(group='SEED_MODALITY_CENTER_HAM',
               project="final_multimodal_training",  config=config)
    wandb_logger = WandbLogger()

    model = ModalityCenterModel(learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay,
                                alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout, seed=seed)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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
        dirpath=os.path.join(
            CHECKPOINT_DIR, 'multi_loss/modality_center/full'),
        filename=dt_string+'HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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
    wandb.finish()


def test_modality_center(seed, config):
    print('YOU ARE RUNNING test modality center FOR HAM DATASET')
    print(config)

    wandb.init(group='TEST_MODALITY_CENTER',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = ModalityCenterModel(learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay,
                                alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout, seed=seed)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed)
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
                 ckpt_path=wandb.config.checkpoint_modality_center)
    wandb.finish()


def test_film(seed, config):
    print('YOU ARE RUNNING FILM FOR HAM DATASET')
    print(config)

    wandb.init(group='TEST_FILM',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = FilmModel(learning_rate=wandb.config.learning_rate,
                      weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed)
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
                 ckpt_path=wandb.config.checkpoint_film)
    wandb.finish()


def main_contrastive_loss(seed, config=None):
    '''
    main function to run the multimodal architecture with contrastive loss
    '''

    print('YOU ARE RUNNING MULTIMODAL NETWORK WITH CONTRASTIVE LOSS FOR HAM DATASET')
    print(config)

    wandb.init(group='SEED_CONTRASTIVE_LOSS_RANDOM_INIT_HAM',
               project="final_multimodal_training",  config=config)
    wandb_logger = WandbLogger()

    model = HamContrastiveModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
    data.set_contrastive_loss_dataloader()
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
        dirpath=os.path.join(
            CHECKPOINT_DIR, 'contrastive_loss/seed/concat/random_init'),
        filename=dt_string+'_HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
        str(wandb.config.weight_decay)+'-{epoch:03d}',
        monitor='val_epoch_loss',
        save_top_k=wandb.config.max_epochs,
        mode='min')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=True)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    wandb.finish()


def main_supervised_contrastive_weights(seed, config=None):
    '''
    main function to run the supervised multimodal architecture with contrastive weights
    this function is only for the CORRELATION setup
    '''
    if config['correlation']:
        text = 'CORR'
    else:
        text = 'CONCAT'

    print('YOU ARE RUNNING SUPERVISED MULTIMODAL (', text,
          ') WITH CONTRASTIVE LEARNING WEIGHTS FOR HAM DATASET')
    print(config)

    wandb.init(group='SEED_SUPERVISED_CONTRAST_WEIGHTS_RANDOM_CONCAT_INIT_' + text,
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.contrastive_checkpoint
    checkpoint = checkpoints[str(seed)]

    # get the cintrastive model from the checkpoint
    cont_model = HamContrastiveModel.load_from_checkpoint(checkpoint)
    # get the supervised model
    model = SupervisedModel(learning_rate=wandb.config.learning_rate,
                            weight_decay=wandb.config.weight_decay, correlation=wandb.config.correlation)
    # initialize the supervised model weights using contrastive model
    model.resnet = cont_model.resnet
    model.resnet.fc = cont_model.resnet.fc
    model.fc1 = cont_model.fc1
    model.fc2 = cont_model.fc2
    model.fc3 = cont_model.fc3
    model.fc4 = cont_model.fc4
    model.fc5 = cont_model.fc5
    model.fc6 = cont_model.fc6
    model.fc7 = cont_model.fc7

    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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
        dirpath=os.path.join(
            CHECKPOINT_DIR, 'supervised/contrastive_init/random_contrastive_init/seed/'+text),
        filename=dt_string+'_HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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

    wandb.finish()


def main_contrastive_evaluate(seed, config=None):
    '''
    main function to run the supervised multimodal architecture with contrastive weights
    weights are frozen
    so the model checks the performance only
    '''
    if config['correlation']:
        text = 'CORR'
    else:
        text = 'CONCAT'

    print('YOU ARE RUNNING SUPERVISED MULTIMODAL (', text,
          ') WITH CONTRASTIVE LEARNING FROZEN WEIGHTS FOR HAM DATASET')
    print(config)

    wandb.init(group='SEED_CONTRASTIVE_RANDOM_INIT_EVAL_' + text,
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.contrastive_checkpoint
    checkpoint = checkpoints[str(seed)]

    # get the cintrastive model from the checkpoint
    cont_model = HamContrastiveModel.load_from_checkpoint(checkpoint)
    # get the supervised model
    model = SupervisedModel(learning_rate=wandb.config.learning_rate,
                            weight_decay=wandb.config.weight_decay, correlation=wandb.config.correlation)
    # initialize the supervised model weights using contrastive model
    model.resnet = cont_model.resnet
    # model.resnet.fc = cont_model.resnet.fc
    model.fc1 = cont_model.fc1
    model.fc2 = cont_model.fc2
    model.fc3 = cont_model.fc3
    model.fc4 = cont_model.fc4
    model.fc5 = cont_model.fc5
    model.fc6 = cont_model.fc6
    model.fc7 = cont_model.fc7

    # freeze network weights (uncomment if you want to freeze the network weights)
    # Freeze the weights of the ResNet-18
    for param in model.resnet.parameters():
        param.requires_grad = False
    model.fc1.requires_grad_(False)
    model.fc2.requires_grad_(False)
    model.fc3.requires_grad_(False)
    model.fc4.requires_grad_(False)
    model.fc5.requires_grad_(False)
    model.fc6.requires_grad_(False)
    model.fc7.requires_grad_(False)

    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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
        dirpath=os.path.join(
            CHECKPOINT_DIR, 'supervised/contrastive_random_init/frozen/seed/'+text),
        filename=dt_string+'_HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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

    wandb.finish()


def main_triplet_center_cross_entropy(seed, config=None):
    '''
    main function to run the multimodal architecture with triplet + center + cross entropy loss
    '''

    print('YOU ARE RUNNING MULTIMODAL NETWORK WITH TRIPLET + CENTER + CROSS ENT. LOSS FOR HAM DATASET')
    print('THIS MODEL IS ALWAYS WITH CORRELATION')
    print(config)

    wandb.init(group='TRIPLET_CENTER_CROSS_LOSS_HAM',
               project="final_multimodal_training",  config=config)
    wandb_logger = WandbLogger()

    model = TripletCenterModel(
        seed, learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, alpha_triplet=wandb.config.alpha_triplet)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
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
        dirpath=os.path.join(
            CHECKPOINT_DIR, 'triplet_center_cross/training'),
        filename=dt_string+'HAM_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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
    wandb.finish()


if __name__ == '__main__':

    # set the seed of the environment
    # Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    # seed_everything(SEED, workers=True)
    # torch.manual_seed(SEED)
    # np.random.seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.use_deterministic_algorithms(True)

    # run baseline
    # main_baseline(supervised_config)

    # run resnet baseline
    # main_resnet(supervised_config)

    # run tabular baseline
    # main_tabular(tabular_config)

    # run multimodal
    # main_supervised_multimodal(supervised_config)

    # run multiloss model (center + cross entropy + triplet)
    # main_multiloss(multiloss_config)

    # run grid search
    # run_grid_search('multi_loss')

    # run daft
    # main_daft(supervised_config)

    # RUN MODELS FOR EVERY SEED
    # for seed in seed_list:
    #     seed_everything(seed, workers=True)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.use_deterministic_algorithms(True)
    #     main_multiloss(seed, multiloss_config)

    # for seed in seed_list:
    #     seed_everything(seed, workers=True)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.use_deterministic_algorithms(True)
    #     main_supervised_multimodal(seed, supervised_config)

    # for seed in seed_list:
    #     seed_everything(seed, workers=True)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.use_deterministic_algorithms(True)
    #     main_tabular(seed, tabular_config)

    # for seed in seed_list:
    #     seed_everything(seed, workers=True)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.use_deterministic_algorithms(True)
    #     main_new_center(seed, multiloss_config)

    # for seed in seed_list:
    #     seed_everything(seed, workers=True)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.use_deterministic_algorithms(True)
    #     main_daft(seed, supervised_config)

    # for seed in [473]:
    #     seed_everything(seed, workers=True)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.use_deterministic_algorithms(True)
    #     main_contrastive_loss(seed, contrastive_loss_config)

    # for seed in seed_list:
    #     seed_everything(seed, workers=True)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.use_deterministic_algorithms(True)
    #     main_contrastive_loss(seed, supervised_config)

    for seed in seed_list:
        seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        main_triplet_center_cross_entropy(seed, triplet_center_config)

    # for seed in seed_list:
    #     seed_everything(seed, workers=True)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.use_deterministic_algorithms(True)
    #     main_supervised_contrastive_weights(seed, supervised_config)

    # for seed in seed_list:
    #     seed_everything(seed, workers=True)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.use_deterministic_algorithms(True)
    #     main_contrastive_evaluate(seed, supervised_config)

    # TESTING
    # test_supervised_multimodal(seed=1997, config=supervised_config)
    # test_supervised_corr_multimodal(seed=1997, config=supervised_config)
    # test_resnet(supervised_config)
    # test_multiloss(seed=1997, config=multiloss_config)
    # test_new_center(seed=1997, config=multiloss_config)
    # test_daft(seed=1997, config=supervised_config)
    # test_tabular(seed=1997, config=tabular_config)
    # test_film(seed=1997, config=supervised_config)
    # test_modality_center(seed=1997, config=multiloss_config)
