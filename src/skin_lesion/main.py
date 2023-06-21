import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from ham_settings import csv_dir, CHECKPOINT_DIR, seed_list, config, SEED
from models.ham_supervised_model import SupervisedModel
from torch import nn
from models.previous.image_model import BaselineModel
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
from models.ham_cross_modal_center import CrossModalCenterModel
from models.ham_modality_specific_center import ModalityCenterModel
from models.previous.ham_contrastive_pretrain_model import HamContrastiveModel
from models.ham_triplet_center_cross_ent import TripletCenterModel
from models.previous.ham_contrastive_center_cross_ent_model import HamContrastiveCenterCrossModel
from models.previous.multiloss import deneme


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

    wandb.init(group='RESNET',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = ResnetModel(learning_rate=wandb.config.learning_rate,
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
        dirpath=os.path.join(CHECKPOINT_DIR, '_RESNET/train'),
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


def test_resnet(seed, config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR RESNET MODEL FOR HAM DATASET')

    wandb.init(group='TEST_RESNET',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    checkpoint = checkpoints[str(seed)]

    # get the model
    model = ResnetModel(learning_rate=wandb.config.learning_rate,
                        weight_decay=wandb.config.weight_decay)
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
                 ckpt_path=checkpoint)
    wandb.finish()


def main_tabular(seed, config=None):
    '''
    main function to run the baseline model
    '''

    print('YOU ARE RUNNING TABULAR MODEL FOR HAM DATASET')
    print(config)

    wandb.init(group='TABULAR',
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
        dirpath=os.path.join(CHECKPOINT_DIR, '_TABULAR/train'),
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

    wandb.init(group='SUPERVISED_CONCAT',
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
            CHECKPOINT_DIR, '_SUPERVISED/train/'),
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

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    wandb.init(group='TEST_SUPERVISED_'+corr,
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    if config['correlation']:
        checkpoints = wandb.config.checkpoint_correlation
    checkpoint = checkpoints[str(seed)]

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
                 ckpt_path=checkpoint)
    wandb.finish()


def main_multiloss(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING SEED MULTI LOSS MODEL CENTER + CROSS ENTROPY LOSSES FOR HAM DATASET')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    wandb.init(group='(NO-LR)CENTER_CROSS_ENT_'+corr,
               project="final_multimodal_training",  config=config)
    wandb_logger = WandbLogger()

    model = MultiLossModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout, seed=seed, correlation=wandb.config.correlation)
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'CENTER_CROSS_ENT/train/'+corr),
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


def test_multiloss(seed, config=None):
    '''
    main function to run the test loop for multiloss architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR MULTILOSS FOR HAM DATASET')

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    wandb.init(group='DENEME_CENTER_CROSS_ENT_'+corr,
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint_concat
    if config['correlation']:
        checkpoints = wandb.config.checkpoint_correlation
    checkpoint = checkpoints[str(seed)]

    # get the model
    # CONCAT
    model = MultiLossModel(seed=SEED,
                           learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed=SEED)
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
                 ckpt_path=checkpoint)
    wandb.finish()


def run_grid_search(network):

    print('YOU ARE RUNNING GRID SEARCH FOR: ', network)

    sweep_config = {
        'method': 'grid',
        'metric': {'goal': 'maximize', 'name': 'val_macro_acc'},
        'parameters': {
            'network': {'value': network},
            'batch_size': {'value': 512},
            'max_epochs': {'value': 10},
            'age': {'value': None},
            'learning_rate': {'values': [1e-4]},
            'weight_decay': {'value': 0},
            'alpha_center': {'values': [0.01, 0.05, 0.1]},
            'alpha_triplet': {'values': [0.2, 0.3, 0.4]},
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
    wandb.agent(sweep_id, function=main_triplet_center_cross_entropy)
    wandb.finish()


def main_daft(seed, config=None):
    '''
    main function to run the supervised multimodal DAFT architecture
    '''

    print('YOU ARE RUNNING DAFT FOR HAM DATASET')
    print(config)

    wandb.init(group='DAFT_PAPER',
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
        dirpath=os.path.join(CHECKPOINT_DIR, '_DAFT/training/PAPER'),
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


def test_daft(seed, config):
    print('YOU ARE RUNNING DAFT FOR HAM DATASET')
    print(config)

    wandb.init(group='TEST_DAFT',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    checkpoint = checkpoints[str(seed)]

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
                 ckpt_path=checkpoint)
    wandb.finish()


def main_cross_modal_center(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING SEED MULTI LOSS MODEL CENTER + CROSS ENTROPY LOSSES FOR HAM DATASET')
    print(config)

    wandb.init(group='CROSS_MODAL_CENTER',
               project="final_multimodal_training",  config=config)
    wandb_logger = WandbLogger()

    model = CrossModalCenterModel(learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay,
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'CROSS_MODAL_CENTER/training'),
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


def test_cross_modal_center(seed, config=None):
    '''
    main function to run the test loop for multiloss architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR MULTILOSS FOR HAM DATASET')

    wandb.init(group='TEST_CROSS_MODAL_CENTER',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    checkpoint = checkpoints[str(seed)]
    # get the model
    # CONCAT
    model = CrossModalCenterModel(seed=seed,
                                  learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, dropout_rate=wandb.config.dropout)

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
                 ckpt_path=checkpoint)
    wandb.finish()


def test_tabular(seed, config):

    print('YOU ARE RUNNING TABULAR MODEL FOR HAM DATASET')
    print(config)

    wandb.init(group='TEST_TABULAR',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    checkpoint = checkpoints[str(seed)]

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
                 ckpt_path=checkpoint)
    wandb.finish()


def main_film(seed, config):
    '''
    main function to run the supervised multimodal DAFT architecture
    '''

    print('YOU ARE RUNNING FILM FOR HAM DATASET')
    print(config)

    wandb.init(group='FILM',
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
        dirpath=os.path.join(CHECKPOINT_DIR, '_FILM/train'),
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


def test_film(seed, config):
    print('YOU ARE RUNNING FILM FOR HAM DATASET')
    print(config)

    wandb.init(group='TEST_FILM',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    checkpoint = checkpoints[str(seed)]

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
                 ckpt_path=checkpoint)
    wandb.finish()


def main_modality_center(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING SEED MULTI LOSS MODEL CENTER + CROSS ENTROPY LOSSES FOR HAM DATASET')
    print(config)

    wandb.init(group='MODALITY_CENTER',
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
            CHECKPOINT_DIR, 'MODALITY_CENTER/training'),
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

    wandb.init(group='TEST_MODALITY_SPEC_CENTER',
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    checkpoint = checkpoints[str(seed)]

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
                 ckpt_path=checkpoint)
    wandb.finish()


def main_contrastive_pretrain(seed, config=None):
    '''
    main function to run the multimodal architecture with contrastive loss
    '''

    print('YOU ARE RUNNING CONTRASTIVE PRETRAINING FOR HAM DATASET')
    print(config)

    wandb.init(group='CONTRASTIVE_PRETRAIN',
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
            CHECKPOINT_DIR, 'CONTRASTIVE_PRETRAIN/training'),
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


def main_contrastive_center_cross_ent(seed, config=None):
    '''
    main function to run the multimodal architecture with contrastive loss
    '''

    print('YOU ARE RUNNING CONTRASTIVE PRETRAINING FOR HAM DATASET')
    print(config)

    wandb.init(group='CONTRASTIVE_CENTER_CROSS_ENT',
               project="final_multimodal_training",  config=config)
    wandb_logger = WandbLogger()

    model = HamContrastiveCenterCrossModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_contrastive=wandb.config.alpha_contrastive, alpha_center=wandb.config.alpha_center, correlation=wandb.config.correlation, seed=seed)
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
            CHECKPOINT_DIR, 'CONTRASTIVE_CENTER_CE/training'),
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


def main_triplet_center_cross_entropy(seed, config=None):
    '''
    main function to run the multimodal architecture with triplet + center + cross entropy loss
    '''

    print('YOU ARE RUNNING MULTIMODAL NETWORK WITH TRIPLET + CENTER + CROSS ENT. LOSS FOR HAM DATASET')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    wandb.init(group='TRIPLET_'+corr,
               project="final_multimodal_training",  config=config)
    wandb_logger = WandbLogger()

    model = TripletCenterModel(
        seed, learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, alpha_triplet=wandb.config.alpha_triplet, correlation=wandb.config.correlation)
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
            CHECKPOINT_DIR, 'TRIPLET', corr, dt_string, 'training'),
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


def test_triplet_center_cross_ent(seed, config):
    print('YOU ARE RUNNING TRIPLET MODEL FOR HAM DATASET')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    wandb.init(group='TEST_TRIPLET_'+corr,
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    checkpoint = checkpoints[str(seed)]

    # get the model
    model = TripletCenterModel(
        seed, learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, alpha_center=wandb.config.alpha_center, alpha_triplet=wandb.config.alpha_triplet, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_data(seed)
    data.set_triplet_dataloader()
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
                 ckpt_path=checkpoint)
    wandb.finish()


def main_finetune(seed, config, percent, zero_shot):
    '''
    main function to run the test loop for TRIPLET MODEL 
    '''
    print('YOU ARE RUNNING ZERO SHOT LEARNING FOR TRIPLET MODEL ')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    name = 'SEMI_SUPER_'
    if zero_shot:
        name = 'ZERO_SHOT_'

    run = wandb.init(group=name+'FINETUNE_'+corr,
                     project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    checkpoint = checkpoints[str(seed)]
    wandb.log({'checkpoint': checkpoint})
    wandb.log({'train_dataset_percent': percent})

    # get the model
    model_ckpt = TripletCenterModel.load_from_checkpoint(checkpoint)
    model = SupervisedModel(learning_rate=wandb.config.learning_rate,
                            weight_decay=wandb.config.weight_decay, correlation=wandb.config.correlation)
    model.resnet = model_ckpt.resnet
    model.fc1 = model_ckpt.fc1
    model.fc2 = model_ckpt.fc2
    model.fc3 = model_ckpt.fc3
    model.fc4 = model_ckpt.fc4
    model.fc5 = model_ckpt.fc5
    model.fc6 = model_ckpt.fc6
    model.fc7 = model_ckpt.fc7
    # change the final classification layers of the model
    model.fc8 = nn.Linear(32, 7)

    if zero_shot:
        # freeze layers
        layers = ['resnet', 'fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7']
        for layer, param in model.named_parameters():
            if param.requires_grad and layer.split('.')[0] in layers:
                print('Layer: ', layer, ' is frozen')
                param.requires_grad = False

    wandb.watch(model, log="all")

    # load the data
    data = HAMDataModule(
        csv_dir, age=wandb.config.age, batch_size=wandb.config.batch_size)
    data.prepare_zero_shot_data(seed=seed, percent=percent)
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
    filename_prefix = dt_string + '_HAM_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(CHECKPOINT_DIR,  'FINETUNE', name,
                           corr, dt_string, 'train')
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_prefix + '-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=10,
        mode='max'
    )

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=10, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=True)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix)


def test_finetune(seed, config, percent, finetune_method, checkpoints):
    print('YOU ARE RUNNING TEST FINETUNE FOR HAM DATASET')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    wandb.init(group='TEST_FINETUNE_'+finetune_method+corr,
               project="final_multimodal_training", config=config)
    wandb_logger = WandbLogger()

    checkpoint = checkpoints[str(seed)]
    wandb.log({'checkpoints': checkpoints})
    wandb.log({'checkpoint': checkpoint})
    wandb.log({'train_dataset_percent': percent})

    # get the model
    model = SupervisedModel(learning_rate=wandb.config.learning_rate,
                            weight_decay=wandb.config.weight_decay, correlation=wandb.config.correlation)
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
                 ckpt_path=checkpoint)
    wandb.finish()


def main_deneme(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING SEED MULTI LOSS MODEL CENTER + CROSS ENTROPY LOSSES FOR HAM DATASET')
    print(config)

    wandb.init(group='deneme_modal_spec',
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


def get_filenames(results):
    print(results)
    filenames = list(results)

    avg_best_epoch_filenames = [
        name + f'-epoch={9:03d}.ckpt'
        for name
        in filenames
    ]
    print(avg_best_epoch_filenames)
    with open("best_epoch_checkpoints.txt", "a") as f:
        content = ", ".join(avg_best_epoch_filenames) + "\n\n"
        f.write(content)

    return avg_best_epoch_filenames


def main(percent, zero_shot, training, **kwargs):
    results = []
    for seed in seed_list:
        seed_everything(SEED, workers=True)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.use_deterministic_algorithms(True)

        if training:
            result = main_finetune(
                seed, config=config['triplet_center_config'], percent=percent, zero_shot=zero_shot)
            results.append(result)
        else:
            test_finetune(seed, config=config['triplet_center_config'], percent=percent, finetune_method=kwargs.get(
                'finetune_method', None), checkpoints=kwargs.get(
                'checkpoints', None))

    if training:
        # get the filenames
        avg_best_epoch_filenames = get_filenames(results)
        new_checkpoints = {
            str(seed): file for seed, file in zip(seed_list, avg_best_epoch_filenames)
        }

        main(percent=percent, zero_shot=zero_shot, training=False, finetune_method=kwargs.get(
            'finetune_method', None), checkpoints=new_checkpoints)


if __name__ == '__main__':

    percents = [0.03, 0.1, 0.03, 0.1]
    zero_shots = [False, False, True, True]
    training = True
    finetune_methods = ['SEMI_SUPERVISED_',
                        'SEMI_SUPERVISED_', 'ZERO_SHOT_', 'ZERO_SHOT_']

    for percent, zero_shot, finetune_method in zip(percents, zero_shots, finetune_methods):
        main(percent, zero_shot, training, finetune_method=finetune_method)

    # set the seed of the environment
    # Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    # SEED = 1997
    # seed_everything(SEED, workers=True)
    # torch.manual_seed(SEED)
    # np.random.seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.use_deterministic_algorithms(True)

    # run_grid_search('triplet_center_cross_entropy')

    # RUN MODELS FOR EVERY SEED

    # for seed in seed_list:
    #     seed_everything(SEED, workers=True)
    #     torch.manual_seed(SEED)
    #     np.random.seed(SEED)
    #     torch.cuda.manual_seed(SEED)
    #     random.seed(SEED)
    #     np.random.seed(SEED)
    #     torch.use_deterministic_algorithms(True)

    #     # main_film(seed, config['film_config'])
    #     # main_supervised_multimodal(seed, config['supervised_config'])
    #     # main_resnet(seed, config['resnet_config'])
    #     # main_tabular(seed, config['tabular_config'])
    #     # main_daft(seed, config['daft_config'])
    #     # main_triplet_center_cross_entropy(
    #     #     seed, config['triplet_center_config'])  # 1) runtriület with concatenation
    #     main_triplet_finetune(
    #         seed, config=config['triplet_center_config'], percent=0.01)

    #############################  ABLATION  ###########################

    # main_multiloss(seed, config['multiloss_config']) # 2) run multiloss model without any lr scheduler
    # main_cross_modal_center(seed, config['cross_modal_center_config'])
    # main_modality_center(seed, config['modality_center_config'])
    # main_contrastive_pretrain(seed, config['contrastive_pretrain_config'])
    # main_contrastive_center_cross_ent(
    #     seed, config=config['contrastive_center_cross_config'])

    ###########################  TEST  ##################################
    # test_resnet(seed, config=config['resnet_config'])
    # test_tabular(seed, config=config['tabular_config'])
    # test_supervised_multimodal(seed, config['supervised_config'])  # CONCAT
    # test_daft(seed, config['daft_config'])
    # test_film(seed, config['film_config'])
    # test_triplet_center_cross_ent(seed, cinfig['triplet_center_config']

    #########################  TEST - ABLATION  ##############################

    # test_multiloss(seed, config['multiloss_config'])  # CONCAT + CORR
    # test_cross_modal_center(seed, config['cross_modal_center_config'])
    # test_modality_center(seed, config['modality_center_config'])
    # test_supervised_multimodal(seed, config['supervised_config'])  # CORRELATION
    # ADD CONTRASTIVE PRETRAIN
