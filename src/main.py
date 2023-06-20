import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
import torchmetrics
import random
from adni_dataset import AdniDataModule

from conv3D.model import AdniModel
from models.previous.contrastive_learning_model import ContrastiveModel
from models.resnet_model import ResNetModel
from models.supervised_model import MultiModModel
from models.tabular_model import TabularModel
from models.triplet_model import TripletModel
from models.daft_model import DaftModel
from models.film_model import FilmModel
from models.modality_specific_model import ModalitySpecificCenterModel
from models.cross_modal_center_loss import CrossModalCenterModel
from models.center_loss_model import CenterLossModel

from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from adni_settings import CSV_DIR, CHECKPOINT_DIR, seed_list, config
import torch.multiprocessing
from datetime import datetime
# from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from lightning.pytorch.tuner import Tuner
import time
# THIS FUNCTION IS NOT BEING USED
# IT WILL BE DELETED LATER


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


def main_tabular(seed, config=None):
    '''
    main function to run the only tabular architecture
    '''

    print('YOU ARE RUNNING ADNI TABULAR')
    print(config)

    run = wandb.init(
        group='MMSE_TABULAR', project='adni_final_results', config=config)
    wandb_logger = WandbLogger()

    wandb.log({"seed": seed})
    # get the model
    model = TabularModel(learning_rate=wandb.config.learning_rate,
                         weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
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
    filename_prefix = dt_string + '_ADNI_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(CHECKPOINT_DIR, 'TABULAR', dt_string, 'train')
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_prefix + '-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max'
    )

# Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=True)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    loss_history_by_epoch = get_loss_history_from_wandb(run)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix), loss_history_by_epoch


def test_tabular(seed, config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR RESNET MODEL FOR HAM DATASET')

    wandb.init(group='MMSE_TEST_TABULAR_FINAL',
               project="adni_final_results", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    wandb.log({'checkpoints': checkpoints})
    checkpoint = checkpoints[str(seed)]

    # get the model
    model = TabularModel(learning_rate=wandb.config.learning_rate,
                         weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size)
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


def main_resnet(seed, config=None):
    '''
    main function to run the resnet architecture
    '''
    print('YOU ARE RUNNING RESNET')
    print(config)

    run = wandb.init(
        group='RESNET', project='adni_final_results', config=config)
    wandb_logger = WandbLogger()

    wandb.log({"seed": seed})

    # get the model
    model = ResNetModel(learning_rate=wandb.config.learning_rate,
                        weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
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
    filename_prefix = dt_string + '_ADNI_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(CHECKPOINT_DIR, 'RESNET', dt_string, 'train')
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_prefix + '-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max'
    )


# Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=False)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    loss_history_by_epoch = get_loss_history_from_wandb(run)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix), loss_history_by_epoch


def test_resnet(seed, config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR RESNET MODEL FOR HAM DATASET')

    wandb.init(group='TEST_RESNET_FINAL',
               project="adni_final_results", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    wandb.log({'checkpoints': checkpoints})
    checkpoint = checkpoints[str(seed)]

    # get the model
    model = ResNetModel(learning_rate=wandb.config.learning_rate,
                        weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size)
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


def main_supervised_multimodal(seed, config=None):
    '''
    main function to run the supervised multimodal architecture
    '''

    print('YOU ARE RUNNING SUPERVISED MULTIMODAL')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    run = wandb.init(group='MMSE_SUPERVISED_'+corr,
                     project='adni_final_results', config=config)
    wandb_logger = WandbLogger()

    wandb.log({"seed": seed})
    # get the model
    model = MultiModModel(learning_rate=wandb.config.learning_rate,
                          weight_decay=wandb.config.weight_decay, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

   # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
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
    filename_prefix = dt_string + '_ADNI_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(CHECKPOINT_DIR, '_SUPERVISED',
                           corr, dt_string, 'train')
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_prefix + '-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max'
    )

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=False)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    loss_history_by_epoch = get_loss_history_from_wandb(run)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix), loss_history_by_epoch


def test_supervised_multimodal(seed, config=None):
    '''
    main function to run the test loop for CROSS ENTROPY architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR CROSS ENTROPY MODEL')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    wandb.init(group='TEST_MMSE_SUPERVISED_'+corr+'_FINAL',
               project="adni_final_results", config=config)
    wandb_logger = WandbLogger()

    # checkpoints = wandb.config.checkpoint_concat
    # if config['correlation']:
    #     checkpoints = wandb.config.checkpoint_corr
    checkpoints = wandb.config.checkpoint
    wandb.log({'checkpoints': checkpoints})
    checkpoint = checkpoints[str(seed)]

    # get the model
    model = MultiModModel(learning_rate=wandb.config.learning_rate,
                          weight_decay=wandb.config.weight_decay, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size)
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


# def main_kfold_multimodal(wandb, wandb_logger, fold_number=2, learning_rate=1e-3, batch_size=8, max_epochs=60, age=None):
#     '''
#     main function to run the multimodal architecture with cross validation
#     '''

#     # path to the csv file
#     # this csv file contains image ids, patient ids and tabular info
#     csv_dir = data_module = KfoldMultimodalDataModule(
#         csv_dir, fold_number=fold_number, age=age, batch_size=batch_size)
#     # get dataloaders for every fold
#     train_dataloaders, val_dataloaders = data_module.prepare_data()

#     train_fold_losses = []
#     val_fold_losses = []
#     accuracies = []

#     accelerator = 'cpu'
#     devices = None
#     if torch.cuda.is_available():
#         accelerator = 'gpu'
#         devices = 1

#     fold_num = 0
#     # train the mdoel
#     for train_dataloader, val_dataloader in zip(train_dataloaders, val_dataloaders):
#         # get the model
#         model = MultiModModel(learning_rate=learning_rate)
#         trainer = Trainer(accelerator=accelerator, devices=devices,
#                           max_epochs=max_epochs, logger=wandb_logger, deterministic=True)
#         trainer.fit(model, train_dataloader, val_dataloader)

#         # log the loss of the fold
#         wandb.log(
#             {"train_fold_loss": model.metrics['train_epoch_losses'][-1], "fold": fold_num})
#         wandb.log(
#             {"val_fold_loss": model.metrics['val_epoch_losses'][-1], "fold": fold_num})

#         # print the fold losses
#         print(
#             {'Fold {fold_num}, final train fold loss': model.metrics['train_epoch_losses'][-1]})
#         print(
#             {'Fold {fold_num}, final val fold loss': model.metrics['val_epoch_losses'][-1]})

#         # add the final val and train losses to the list
#         train_fold_losses.append(model.metrics['train_epoch_losses'][-1])
#         val_fold_losses.append(model.metrics['val_epoch_losses'][-1])
#         # accuracies.append(model.metrics["train_accuracy"][-1])

#         # print(model.metrics['valid_accuracy'])
#         # wandb.log({'Valid acc final': model.metrics['valid_accuracy'][-1]})
#         # wandb.log({"Train acc final": model.metrics["train_accuracy"][-1]})

#         fold_num += 1

#     print('all the train losses: ', train_fold_losses)
#     print('all the val losses: ', val_fold_losses)

#     # log the average loss of folds
#     wandb.log({"Average fold loss (val)": sum(
#         val_fold_losses)/len(val_fold_losses)})
#     wandb.log({"Average fold loss (train)":  sum(
#         train_fold_losses)/len(train_fold_losses)})

#     # wandb.log({"Valid acc avg": sum(
#     #     model.metrics['valid_accuracy'])/len(model.metrics['valid_accuracy'])})
#     # wandb.log({"Train acc avg":  sum(
#     # model.metrics['train_accuracy'])/len(model.metrics['train_accuracy'])})
#     # wandb.log({"Mean score":scores.mean()})


# def main_contrastive_learning(config=None):
#     '''
#     main function to run the multimodal architecture
#     '''

#     print('YOU ARE RUNNING CONTRASTIVE MODEL')
#     print(config)

#     wandb.init(group='CONTRASTIVE', project='adni_final_results', config=config)
#     wandb_logger = WandbLogger()

#     # get the model
#     model = ContrastiveModel(
#         learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
#     wandb.watch(model, log="all")

#     # load the data
#     data = AdniDataModule(batch_size=wandb.config.batch_size,
#                           spatial_size=wandb.config.spatial_size)
#     data.prepare_data()
#     data.set_contrastive_loss_dataloader()
#     train_dataloader = data.train_dataloader()
#     val_dataloader = data.val_dataloader()

#     accelerator = 'cpu'
#     devices = None
#     if torch.cuda.is_available():
#         accelerator = 'gpu'
#         devices = 1

#     # save the checkpoint in a different folder
#     # use datetime value in the file name
#     date_time = datetime.now()
#     dt_string = date_time.strftime("%d.%m.%Y-%H.%M")
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=os.path.join(CHECKPOINT_DIR, 'contrastive'), filename='lr='+str(wandb.config.learning_rate)+'_wd='+str(wandb.config.weight_decay)+'_'+dt_string+'-{epoch:03d}')

#     # Add learning rate scheduler monitoring
#     lr_monitor = LearningRateMonitor(logging_interval='epoch')
#     trainer = Trainer(accelerator=accelerator, devices=devices,
#                       max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=10)
#     trainer.fit(model, train_dataloaders=train_dataloader,
#                 val_dataloaders=val_dataloader)

def main_modality_specific_center(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING MODALITY SPECIFIC CENTER MODEL')
    print(config)

    run = wandb.init(group='MODALITY SPECIFIC',
                     project='adni_final_results', config=config)
    wandb_logger = WandbLogger()

    wandb.log({"seed": seed})
    # get the model
    model = ModalitySpecificCenterModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, seed=seed, alpha_center=wandb.config.alpha_center)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
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
    filename_prefix = dt_string + '_ADNI_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(
        CHECKPOINT_DIR, 'MODALITY_SPECIFIC', dt_string, 'train')
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_prefix + '-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max'
    )

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=False)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    loss_history_by_epoch = get_loss_history_from_wandb(run)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix), loss_history_by_epoch


def test_modality_specific_center(seed, config=None):
    '''
    main function to run the test loop for modality specific center loss architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR MODALITY SPECIFIC CENTER LOSS')

    wandb.init(group='TEST_MODALITY_SPECIFIC',
               project="adni_final_results", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    wandb.log({'checkpoints': checkpoints})
    checkpoint = checkpoints[str(seed)]

    # get the model
    model = ModalitySpecificCenterModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, seed=seed, alpha_center=wandb.config.alpha_center)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size)
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


def main_cross_modal_center(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING CROSS MODAL CENTER MODEL')
    print(config)

    run = wandb.init(group='CROSS MODAL',
                     project='adni_final_results', config=config)
    wandb_logger = WandbLogger()

    wandb.log({"seed": seed})
    # get the model
    model = CrossModalCenterModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, seed=seed, alpha_center=wandb.config.alpha_center)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
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
    filename_prefix = dt_string + '_ADNI_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(CHECKPOINT_DIR, 'CROSS_MODAL', dt_string, 'train')
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_prefix + '-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max'
    )

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=False)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    loss_history_by_epoch = get_loss_history_from_wandb(run)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix), loss_history_by_epoch


def test_cross_modal_center(seed, config=None):
    '''
    main function to run the test loop for CROSS MODAL center architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR CROSS MODAL CENTER LOSS')

    wandb.init(group='TEST_CROSS_MODAL',
               project="adni_final_results", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    wandb.log({'checkpoints': checkpoints})
    checkpoint = checkpoints[str(seed)]

    # get the model
    model = CrossModalCenterModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, seed=seed, alpha_center=wandb.config.alpha_center)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size)
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


def main_center_loss(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING CENTER LOSS MODEL')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    run = wandb.init(group='CETNER LOSS_'+corr,
                     project='adni_final_results', config=config)
    wandb_logger = WandbLogger()

    wandb.log({"seed": seed})
    # get the model
    model = CenterLossModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, seed=seed, alpha_center=wandb.config.alpha_center, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
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
    filename_prefix = dt_string + '_ADNI_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(CHECKPOINT_DIR, 'CENTER_LOSS',
                           corr, dt_string, 'train')
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_prefix + '-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max'
    )

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=False)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    loss_history_by_epoch = get_loss_history_from_wandb(run)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix), loss_history_by_epoch


def test_center_loss(seed, config=None):
    '''
    main function to run the test loop for CENTER LOSS architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR CENTER LOSS')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    wandb.init(group='TEST_CENTER_LOSS_'+corr,
               project="adni_final_results", config=config)
    wandb_logger = WandbLogger()

    # checkpoints = wandb.config.checkpoint_concat
    # if config['correlation']:
    #     checkpoints = wandb.config.checkpoint_corr
    checkpoints = wandb.config.checkpoint
    wandb.log({'checkpoints': checkpoints})
    checkpoint = checkpoints[str(seed)]

  # get the model
    model = CenterLossModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, seed=seed, alpha_center=wandb.config.alpha_center, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size)
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


def main_triplet(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING TRIPLET LOSS MODEL')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    run = wandb.init(group='TRIPLET_'+corr,
                     project='adni_final_results', config=config)
    wandb_logger = WandbLogger()

    wandb.log({"seed": seed})
    # get the model
    model = TripletModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, seed=seed, alpha_center=wandb.config.alpha_center, alpha_triplet=wandb.config.alpha_triplet, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
    data.prepare_data(seed)
    data.set_triplet_loss_dataloader()
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
    filename_prefix = dt_string + '_ADNI_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(CHECKPOINT_DIR,  'TRIPLET',
                           corr, dt_string, 'train')
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_prefix + '-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max'
    )

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=False)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    loss_history_by_epoch = get_loss_history_from_wandb(run)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix), loss_history_by_epoch


def test_triplet(seed, config=None, **kwargs):
    '''
    main function to run the test loop for TRIPLET MODEL 
    '''

    print('YOU ARE RUNNING TEST LOOP FOR TRIPLET MODEL ')
    print(config)

    corr = 'CONCAT'
    if config['correlation']:
        corr = 'CORRELATION'

    wandb.init(group='TEST_'+kwargs.get('finetune_method', '')+'TRIPLET_'+corr,
               project="adni_final_results", config=config)
    wandb_logger = WandbLogger()

    # checkpoints = wandb.config.checkpoint_concat
    # if config['correlation']:
    #     checkpoints = wandb.config.checkpoint_corr
    checkpoints = wandb.config.checkpoint
    wandb.log({'checkpoints': checkpoints})
    checkpoint = checkpoints[str(seed)]
    wandb.log({'checkpoint': checkpoint})
    wandb.log({'train_dataset_percent': kwargs.get('finetune_percent', '')})

    # get the model
    model = TripletModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, seed=seed, alpha_center=wandb.config.alpha_center, alpha_triplet=wandb.config.alpha_triplet, correlation=wandb.config.correlation)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size)
    data.prepare_data(seed=seed)
    data.set_triplet_loss_dataloader()
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


def main_daft(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING DAFT MODEL')
    print(config)

    run = wandb.init(group='DAFT', project='adni_final_results', config=config)
    wandb_logger = WandbLogger()

    wandb.log({"seed": seed})

    # get the model
    model = DaftModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
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
    filename_prefix = dt_string + '_ADNI_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(CHECKPOINT_DIR, 'DAFT', dt_string, 'train')
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_prefix + '-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max'
    )

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=False)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    loss_history_by_epoch = get_loss_history_from_wandb(run)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix), loss_history_by_epoch


def test_daft(seed, config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR RESNET MODEL FOR HAM DATASET')

    wandb.init(group='TEST_DAFT_FINAL',
               project="adni_final_results", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    wandb.log({'checkpoints': checkpoints})
    checkpoint = checkpoints[str(seed)]

    # get the model
    model = DaftModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size)
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
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[lr_monitor], deterministic=False)
    trainer.test(model, dataloaders=test_dataloader,
                 ckpt_path=checkpoint)
    wandb.finish()


def main_film(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING FILM MODEL')
    print(config)

    run = wandb.init(group='FILM', project='adni_final_results', config=config)
    wandb_logger = WandbLogger()

    wandb.log({"seed": seed})

    # get the model
    model = FilmModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
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
    filename_prefix = dt_string + '_ADNI_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(CHECKPOINT_DIR, 'FILM', dt_string, 'train')
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_prefix + '-{epoch:03d}',
        monitor='val_macro_acc',
        save_top_k=wandb.config.max_epochs,
        mode='max'
    )

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=False)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    loss_history_by_epoch = get_loss_history_from_wandb(run)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix), loss_history_by_epoch


def test_film(seed, config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR FILM')

    wandb.init(group='TEST_FILM_FINAL',
               project="adni_final_results", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    wandb.log({'checkpoints': checkpoints})
    checkpoint = checkpoints[str(seed)]

    # get the model
    model = FilmModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size)
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


def main_triplet_finetune(seed, config, percent, checkpoints, zero_shot):
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

    run = wandb.init(group='TRIPLET_'+name+corr,
                     project="adni_final_results", config=config)
    wandb_logger = WandbLogger()

    checkpoint = checkpoints[str(seed)]
    wandb.log({'checkpoint': checkpoint})
    wandb.log({'train_dataset_percent': percent})

    # get the model
    model = TripletModel.load_from_checkpoint(checkpoint)
    # change the final classification layers of the model
    model.fc7 = nn.Linear(32, 3)

    if zero_shot:
        # freeze layers
        layers = ['resnet', 'fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6']
        for layer, param in model.named_parameters():
            if param.requires_grad and layer.split('.')[0] in layers:
                print('Layer: ', layer, ' is frozen')
                param.requires_grad = False

    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size)
    data.prepare_zero_shot_data(seed=seed, percent=percent)
    data.set_triplet_loss_dataloader()
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
    filename_prefix = dt_string + '_ADNI_SEED=' + str(seed) + '_lr=' + str(
        wandb.config.learning_rate) + '_wd=' + str(wandb.config.weight_decay)
    dirpath = os.path.join(CHECKPOINT_DIR,  'TRIPLET', name,
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
                      max_epochs=10, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], deterministic=False)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    loss_history_by_epoch = get_loss_history_from_wandb(run)

    wandb.finish()

    return os.path.join(dirpath, filename_prefix), loss_history_by_epoch


def run_grid_search(network):

    print('YOU ARE RUNNING GRID SEARCH FOR: ', network)

    sweep_config = {
        'method': 'grid',
        'metric': {'goal': 'minimize', 'name': 'val_epoch_loss'},
        'parameters': {
            'network': {'value': network},
            'batch_size': {'value': 32},
            'max_epochs': {'value': 30},
            'age': {'value': None},
            'spatial_size': {'value': (120, 120, 120)},
            'learning_rate': {'values': [0.03, 0.013, 0.0055, 0.0023, 0.001]},
            'weight_decay': {'values': [0, 1e-2, 1e-4]},
        }
    }

    count = len(sweep_config['parameters']['learning_rate']['values']) * \
        len(sweep_config['parameters']['weight_decay']['values'])

    # sweep
    sweep_id = wandb.sweep(
        sweep_config, project="adni_final_results", entity="multimodal_network")
    wandb.agent(sweep_id, function=grid_search, count=count)
    wandb.finish()


def grid_search(config=None):
    '''
    main function to run grid search on the models
    '''
    with wandb.init(config=config):

        config = wandb.config
        wandb_logger = WandbLogger()

        # load the data
        data = AdniDataModule(
            config.batch_size, spatial_size=config.spatial_size)
        data.prepare_data()

        if config.network == 'resnet':
            # get the model
            model = ResNetModel(learning_rate=config.learning_rate,
                                weight_decay=config.weight_decay)
            data.set_resnet_dataset()

        elif config.network == 'tabular':
            # get the model
            model = TabularModel(learning_rate=config.learning_rate,
                                 weight_decay=config.weight_decay)
            data.set_supervised_multimodal_dataloader()

        elif config.network == 'supervised':
            # get the model
            model = MultiModModel(learning_rate=config.learning_rate,
                                  weight_decay=config.weight_decay)
            data.set_supervised_multimodal_dataloader()

        elif config.network == 'contrastive':
            # get the model
            model = ContrastiveModel(learning_rate=config.learning_rate,
                                     weight_decay=config.weight_decay)
            # load the data
            data.set_contrastive_loss_dataloader()

        elif config.network == 'triplet':
            # get the model
            model = TripletModel(learning_rate=config.learning_rate,
                                 weight_decay=config.weight_decay)
            # load the data
            data.set_triplet_loss_dataloader()

        elif config.network == 'daft':
            # get the model
            model = DaftModel(learning_rate=config.learning_rate,
                              weight_decay=config.weight_decay)
            # load the data
            data.set_supervised_multimodal_dataloader()

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


def get_embeddings(wandb, wandb_logger):

    print('YOU ARE RETRIEVING AND VISUALIZING EMBEDDINGS OF: ',
          wandb.config.model)
    print(wandb.config)

    # copy the weights from the checkpoint
    # use triplet model for both triplet loss and contrastive loss because the models are exactly the same
    model = TripletModel.load_from_checkpoint(
        wandb.config.checkpoint, learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

    # set the model to eval mode to run knn
    torch.set_grad_enabled(False)
    model.eval()

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
    data.prepare_data()
    # use supervised multimodal dataset because we only need images and tabular data woth labels (no negative and positive pairs for contrastive learning)
    data.set_supervised_multimodal_dataloader()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    train_encodings = []
    val_encodings = []
    train_labels = []
    val_labels = []

    # load the data as batches to fit it into the memory
    # get the training batches and calculate the encodings
    for step, batch in enumerate(train_dataloader):

        img, tab, label = batch[0], batch[1], batch[2]
        # calculate the encodings
        encodings = model(img, tab)
        # add encodings to the list
        train_encodings.extend(encodings.numpy())
        # add the labels to the list
        train_labels.extend(label.tolist())

    # get the validation batches and calculate the encodings
    for step, batch in enumerate(val_dataloader):

        img, tab, label = batch[0], batch[1], batch[2]
        # calculate the encodings
        encodings = model(img, tab)
        # add encodings to the list
        val_encodings.extend(encodings.numpy())
        # add the labels to the list
        val_labels.extend(label.tolist())

    # generate embedding projection table in wandb
    # Load the dataset
    val_encodings = np.stack(
        val_encodings, axis=0)
    train_encodings = np.stack(
        train_encodings, axis=0)

    val_labels = np.stack(val_labels, axis=0)
    train_labels = np.stack(train_labels, axis=0)

    val_data = pd.DataFrame(val_encodings)
    val_data['target'] = val_labels
    val_data.columns = [str(col) for col in val_data.columns]
    print('val_data: ', val_data)

    train_data = pd.DataFrame(train_encodings)
    train_data['target'] = train_labels
    train_data.columns = [str(col) for col in train_data.columns]

    wandb.log({"val_encodings": wandb.Table(dataframe=val_data)})
    wandb.log({"train_encodings": wandb.Table(dataframe=train_data)})

    return train_encodings, train_labels, val_encodings, val_labels


# def knn(config):

#     print('YOU ARE RUNNING KNN FOR: ',
#           config['model'])
#     print(config)

#     wandb.init(project='adni_final_results', config=config)
#     wandb_logger = WandbLogger()

#     # get the embeddings of the model
#     train_encodings, train_labels, val_encodings, val_labels = get_embeddings(
#         wandb, wandb_logger)

#     # track macro and micro accuracy
#     knn_macro_accuracy = torchmetrics.Accuracy(
#         task='multiclass', average='macro', num_classes=3, top_k=1)
#     knn_micro_accuracy = torchmetrics.Accuracy(
#         task='multiclass', average='micro', num_classes=3, top_k=1)

#     knn = KNeighborsClassifier(n_neighbors=wandb.config.n_neighbors, n_jobs=-1)
#     knn.fit(train_encodings, train_labels)

#     # get predictions
#     pred = knn.predict(val_encodings)

#     # accuracy: (tp + tn) / (p + n)
#     micro_acc = knn_micro_accuracy(
#         torch.tensor(pred), torch.tensor(val_labels))
#     print('Micro Accuracy: %f' % micro_acc)
#     wandb.log({"KNN micro Acc": micro_acc})

#     macro_acc = knn_macro_accuracy(
#         torch.tensor(pred), torch.tensor(val_labels))
#     print('Macro Accuracy: %f' % macro_acc)
#     wandb.log({"KNN macro Acc": macro_acc})

#     # precision tp / (tp + fp)
#     precision = precision_score(val_labels, pred, average='macro')
#     print('Precision macro: %f' % precision)
#     wandb.log({"KNN Precision macro": precision})
#     # recall: tp / (tp + fn)
#     recall = recall_score(val_labels, pred, average='macro')
#     print('Recall macro: %f' % recall)
#     wandb.log({"KNN Recall macro": recall})
#     # f1: 2 tp / (2 tp + fp + fn)
#     f1 = f1_score(val_labels, pred, average='macro')
#     print('F1 score macro: %f' % f1)
#     wandb.log({"KNN F1 Score macro": f1})

def get_loss_history_from_wandb(run, expected_size=40):
    api = wandb.Api()

    for i in range(12):
        time.sleep(5)
        api.flush()
        api_run = api.run(run.entity + "/" + run.project + "/" + run.id)

        loss_history_by_epoch = list(api_run.history()["val_epoch_loss"])
        # Filters out NaN values, because NaN != NaN
        loss_history_by_epoch = [x for x in loss_history_by_epoch if x == x]

        if len(loss_history_by_epoch) == expected_size:
            return loss_history_by_epoch

    raise Exception(
        f"could not retrieve the loss history with correct size within 60 seconds... Excepted size {expected_size}, but got run of size {len(loss_history_by_epoch)}!")


def find_best_epoch(results):
    print(results)
    filenames, loss_histories = list(zip(*results))

    # Check if all five seeds produced loss_histories with equal length
    n_epochs = len(loss_histories[0])
    print('N EPOCHS', n_epochs)
    print('LOSS HISTORIES', loss_histories)
    print('LENGTH OF LOSS HISTORIES 1', loss_histories[1])
    print('LENGTH OF LOSS HISTORIES 2', loss_histories[2])
    print('LENGTH OF LOSS HISTORIES 3', loss_histories[3])
    print('LENGTH OF LOSS HISTORIES 4', loss_histories[4])
    assert all([len(x) == n_epochs for x in loss_histories])

    avg_loss = [sum([l[i] for l in loss_histories]) /
                len(loss_histories) for i in range(n_epochs)]
    min_loss = min(avg_loss)
    min_loss_epoch_index = avg_loss.index(min_loss)
    avg_best_epoch_filenames = [
        name + f'-epoch={min_loss_epoch_index:03d}.ckpt'
        for name
        in filenames
    ]
    print(avg_best_epoch_filenames)
    with open("best_epoch_checkpoints.txt", "a") as f:
        content = ", ".join(avg_best_epoch_filenames) + "\n\n"
        f.write(content)

    return avg_best_epoch_filenames


def get_filenames(results):
    print(results)
    filenames, loss_histories = list(zip(*results))

    avg_best_epoch_filenames = [
        name + f'-epoch={10:03d}.ckpt'
        for name
        in filenames
    ]
    print(avg_best_epoch_filenames)
    with open("best_epoch_checkpoints.txt", "a") as f:
        content = ", ".join(avg_best_epoch_filenames) + "\n\n"
        f.write(content)

    return avg_best_epoch_filenames


def main(model_name="OTHER", run_test_epoch=False, **kwargs):
    # RUN MODELS FOR EVERY SEED
    results = []

    for seed in seed_list:
        seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)

        config_name = ''

        if model_name == "DAFT":
            config_name = 'daft_config'
            result = main_daft(seed, config['daft_config'])

            results.append(result)

        elif model_name == "DAFT_TEST":
            test_daft(seed, config['daft_config'])

        elif model_name == "FILM":
            config_name = 'film_config'
            result = main_film(seed, config['film_config'])

            results.append(result)

        elif model_name == "FILM_TEST":
            test_film(seed, config['film_config'])

        elif model_name == "RESNET":
            config_name = 'resnet_config'
            result = main_resnet(seed, config['resnet_config'])

            results.append(result)

        elif model_name == "RESNET_TEST":
            test_resnet(seed, config['resnet_config'])

        elif model_name == "TABULAR":
            config_name = 'tabular_config'
            result = main_tabular(seed, config['tabular_config'])

            results.append(result)

        elif model_name == "TABULAR_TEST":
            test_tabular(seed, config['tabular_config'])

        elif model_name == "SUPERVISED_CONCAT":
            config_name = 'supervised_config'
            config['supervised_config']['correlation'] = False
            result = main_supervised_multimodal(
                seed, config['supervised_config'])

            results.append(result)

        elif model_name == "SUPERVISED_CONCAT_TEST":
            config['supervised_config']['correlation'] = False
            test_supervised_multimodal(seed, config['supervised_config'])

        elif model_name == "SUPERVISED_CORR":
            config_name = 'supervised_config'
            config['supervised_config']['correlation'] = True
            result = main_supervised_multimodal(
                seed, config['supervised_config'])

            results.append(result)

        elif model_name == "SUPERVISED_CORR_TEST":
            config['supervised_config']['correlation'] = True
            test_supervised_multimodal(seed, config['supervised_config'])

        elif model_name == "TRIPLET_CONCAT":
            config_name = 'triplet_config'
            config['triplet_config']['correlation'] = False
            result = main_triplet(
                seed, config['triplet_config'])

            results.append(result)

        elif model_name == "TRIPLET_CONCAT_TEST":
            config['triplet_config']['correlation'] = False
            test_triplet(seed, config['triplet_config'])

        elif model_name == "FINETUNE_TRIPLET_CORR":
            config_name = 'triplet_config'
            config['triplet_config']['correlation'] = True
            percent = kwargs.get('finetune_percent', None)
            checkpoints = kwargs.get('finetune_checkpoints', None)
            zero_shot = kwargs.get('finetune_zero_shot', None)

            result = main_triplet_finetune(
                seed, config['triplet_config'], checkpoints=checkpoints, zero_shot=zero_shot, percent=percent)

            results.append(result)

        elif model_name == "FINETUNE_TRIPLET_CORR_TEST":
            config['triplet_config']['correlation'] = True
            test_triplet(seed, config['triplet_config'], finetune_method=kwargs.get(
                'finetune_method', None), finetune_percent=kwargs.get('finetune_percent', ''))

        elif model_name == "TRIPLET_CORR":
            config_name = 'triplet_config'
            config['triplet_config']['correlation'] = True
            result = main_triplet(
                seed, config['triplet_config'])

            results.append(result)

        elif model_name == "TRIPLET_CORR_TEST":
            config['triplet_config']['correlation'] = True
            test_triplet(seed, config['triplet_config'])

        elif model_name == "MODALITY_SPECIFIC":
            config_name = 'modality_specific_center_config'
            result = main_modality_specific_center(
                seed, config['modality_specific_center_config'])

            results.append(result)

        elif model_name == "MODALITY_SPECIFIC_TEST":
            test_modality_specific_center(
                seed, config['modality_specific_center_config'])

        elif model_name == "CROSS_MODAL":
            config_name = 'cross_modal_center_config'
            result = main_cross_modal_center(
                seed, config['cross_modal_center_config'])

            results.append(result)

        elif model_name == "CROSS_MODAL_TEST":
            test_cross_modal_center(seed, config['cross_modal_center_config'])

        elif model_name == "CENTER_LOSS_CONCAT":
            config_name = 'center_loss_config'
            config['center_loss_config']['correlation'] = False
            result = main_center_loss(
                seed, config['center_loss_config'])

            results.append(result)

        elif model_name == "CENTER_LOSS_CONCAT_TEST":
            config['center_loss_config']['correlation'] = False
            test_center_loss(seed, config['center_loss_config'])

        elif model_name == "CENTER_LOSS_CORR":
            config_name = 'center_loss_config'
            config['center_loss_config']['correlation'] = True
            result = main_center_loss(
                seed, config['center_loss_config'])

            results.append(result)

        elif model_name == "CENTER_LOSS_CORR_TEST":
            config['center_loss_config']['correlation'] = True
            test_center_loss(seed, config['center_loss_config'])

    if run_test_epoch:
        avg_best_epoch_filenames = find_best_epoch(results)

        config[config_name]["checkpoint"] = {
            seed: file for seed, file in zip(seed_list, avg_best_epoch_filenames)
        }

        main(model_name=model_name + "_TEST", run_test_epoch=False)

    if kwargs.get('test_finetune', False):
        avg_best_epoch_filenames = get_filenames(results)

        config[config_name]["checkpoint"] = {
            seed: file for seed, file in zip(seed_list, avg_best_epoch_filenames)
        }

        main(model_name=kwargs.get(
            'test_finetune_function', None), run_test_epoch=False, finetune_method=kwargs.get('finetune_method', False), finetune_percent=kwargs.get('finetune_percent', ''))


if __name__ == '__main__':
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    model_name = ["FINETUNE_TRIPLET_CORR"]
    print('the models to run', model_name)
    root = r'/vol/aimspace/users/erks/experiments/adni_checkpoints/TRIPLET/CORRELATION'
    checkpoints = {'1997': root+r"/05.06.2023-21.18/train/05.06.2023-21.18_ADNI_SEED=1997_lr=0.01_wd=0-epoch=028.ckpt",
                   '25': root+r"/05.06.2023-23.50/train/05.06.2023-23.50_ADNI_SEED=25_lr=0.01_wd=0-epoch=028.ckpt",
                   '12': root+r"/06.06.2023-04.24/train/06.06.2023-04.24_ADNI_SEED=12_lr=0.01_wd=0-epoch=028.ckpt",
                   '1966': root+r"/06.06.2023-08.58/train/06.06.2023-08.58_ADNI_SEED=1966_lr=0.01_wd=0-epoch=028.ckpt",
                   '3297': root+r"/06.06.2023-10.52/train/06.06.2023-10.52_ADNI_SEED=3297_lr=0.01_wd=0-epoch=028.ckpt"}
    for model in model_name:
        main(model, run_test_epoch=False, test_finetune=True,
             finetune_method='SEMI_SUPERVISED_', test_finetune_function='FINETUNE_TRIPLET_CORR_TEST', finetune_percent=0.01, finetune_checkpoints=checkpoints, finetune_zero_shot=False)
        # main(model, run_test_epoch=False, test_finetune=True,
        #      finetune_method='SEMI_SUPERVISED_', test_finetune_function='FINETUNE_TRIPLET_CORR_TEST', finetune_percent=0.1, finetune_checkpoints=checkpoints, finetune_zero_shot=False)
        # main(model, run_test_epoch=False, test_finetune=True,
        #      finetune_method='ZERO_SHOT_', test_finetune_function='FINETUNE_TRIPLET_CORR_TEST', finetune_percent=0.01, finetune_checkpoints=checkpoints, finetune_zero_shot=True)
        # main(model, run_test_epoch=False, test_finetune=True,
        #      finetune_method='ZERO_SHOT_', test_finetune_function='FINETUNE_TRIPLET_CORR_TEST', finetune_percent=0.1, finetune_checkpoints=checkpoints, finetune_zero_shot=True)

    # model_name = ["SUPERVISED_CONCAT"]
    # print('the models to run', model_name)
    # for model in model_name:
    #     main(model, run_test_epoch=True)
##############################################################################

    # main_resnet(seed, config['resnet_config'])
    # main_tabular(seed, config['tabular_config'])
    # main_daft(seed, config['daft_config'])
    # main_supervised_multimodal(seed, config['supervised_config'])
    # main_film(seed, config['film_config'])
    # main_triplet(seed, config['triplet_center_config'])

    ######################### ABLATION ###########################
    # main_modality_specific_center(
    #     seed, config['modality_specific_center_config'])
    # main_cross_modal_center(
    #     seed, config['cross_modal_center_config'])
    # main_center_loss(seed, config['center_loss_config'])

    ########################## TEST ###############################

    # test_resnet(seed, config['resnet_config'])
    # test_tabular(seed, config['tabular_config'])
    # 1) corr 2) concat
    # test_supervised_multimodal(seed, config['supervised_config'])
    # test_daft(seed, config['daft_config'])
    # test_film(seed, config['film_config'])
    # 1) corr 2) concat
    # test_triplet(seed, config['triplet_center_config'])

    ########################## TEST ABLATION ###############################

    # test_modality_specific_center(
    #     seed, config['modality_specific_center_config'])
    # test_cross_modal_center(
    #     seed, config['cross_modal_center_config'])
    # 1 (concat) # 2 (corr)
    # test_center_loss(seed, config['center_loss_config'])

##############################################################################
