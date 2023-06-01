import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from sklearn.neighbors import KNeighborsClassifier
import torchmetrics
import random
from adni_dataset import AdniDataModule

from conv3D.model import AdniModel
from models.resnet_model import ResNetModel
from models.supervised_model import MultiModModel
from models.tabular_model import TabularModel
from models.contrastive_learning_model import ContrastiveModel
from models.triplet_model import TripletModel
from models.daft_model import DaftModel
from models.multi_loss_model import MultiLossModel
from models.film_model import FilmModel

from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from adni_settings import CSV_DIR, CHECKPOINT_DIR, seed_list, config
import torch.multiprocessing
from datetime import datetime
# from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from lightning.pytorch.tuner import Tuner

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

    wandb.init(group='TABULAR', project='adni_multimodal', config=config)
    wandb_logger = WandbLogger()

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
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'TABULAR/train'),
        filename=dt_string+'_ADNI_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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


def test_tabular(seed, config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR RESNET MODEL FOR HAM DATASET')

    wandb.init(group='TEST_TABULAR',
               project="adni_multimodal", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
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

    wandb.init(group='RESNET', project='adni_multimodal', config=config)
    wandb_logger = WandbLogger()

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
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'RESNET/train'),
        filename=dt_string+'_ADNI_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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


def test_resnet(seed, config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR RESNET MODEL FOR HAM DATASET')

    wandb.init(group='TEST_RESNET',
               project="adni_multimodal", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
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

    wandb.init(group='SUPERVISED_'+corr,
               project='adni_multimodal', config=config)
    wandb_logger = WandbLogger()

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
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            CHECKPOINT_DIR, '_SUPERVISED/' + corr + '/train/'),
        filename=dt_string+'_ADNI_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR RESNET MODEL FOR HAM DATASET')

    wandb.init(group='TEST_SUPERVISED',
               project="adni_multimodal", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
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

#     wandb.init(group='CONTRASTIVE', project='adni_multimodal', config=config)
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


def main_triplet(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING TRIPLET LOSS MODEL')
    print(config)

    wandb.init(group='TRIPLET', project='adni_multimodal', config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = TripletModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, seed=seed, alpha_center=wandb.config.alpha_center, alpha_triplet=wandb.config.alpha_triplet)
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
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            CHECKPOINT_DIR, 'TRIPLET/train'),
        filename=dt_string+'_ADNI_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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


def test_triplet(seed, config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR RESNET MODEL FOR HAM DATASET')

    wandb.init(group='TEST_TRIPLET',
               project="adni_multimodal", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
    checkpoint = checkpoints[str(seed)]

    # get the model
    model = TripletModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay, seed=seed, alpha_center=wandb.config.alpha_center, alpha_triplet=wandb.config.alpha_triplet)
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


def main_multiloss(config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING MULTI LOSS MODEL WITH CENTER + CROSS ENTROPY LOSSES')
    print(config)

    wandb.init(group='multiLoss-adni',
               project='adni_multimodal', config=config)
    wandb_logger = WandbLogger()

    # get the modela
    model = MultiLossModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(batch_size=wandb.config.batch_size,
                          spatial_size=wandb.config.spatial_size)
    data.prepare_data()
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
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'multi_loss'), filename='lr='+str(wandb.config.learning_rate)+'_wd='+str(wandb.config.weight_decay)+'_'+dt_string+'-{epoch:03d}')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def main_daft(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING DAFT MODEL')
    print(config)

    wandb.init(group='DAFT', project='adni_multimodal', config=config)
    wandb_logger = WandbLogger()

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
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'DAFT/train'),
        filename=dt_string+'_ADNI_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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


def test_daft(seed, config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR RESNET MODEL FOR HAM DATASET')

    wandb.init(group='TEST_DAFT',
               project="adni_multimodal", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
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
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[lr_monitor], deterministic=True)
    trainer.test(model, dataloaders=test_dataloader,
                 ckpt_path=checkpoint)
    wandb.finish()


def main_film(seed, config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING FILM MODEL')
    print(config)

    wandb.init(group='FILM', project='adni_multimodal', config=config)
    wandb_logger = WandbLogger()

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
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, 'FILM/train'),
        filename=dt_string+'_ADNI_SEED='+str(seed)+'_lr='+str(wandb.config.learning_rate)+'_wd=' +
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


def test_film(seed, config=None):
    '''
    main function to run the test loop for resnet architecture
    '''

    print('YOU ARE RUNNING TEST LOOP FOR FILM')

    wandb.init(group='TEST_FILM',
               project="adni_multimodal", config=config)
    wandb_logger = WandbLogger()

    checkpoints = wandb.config.checkpoint
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
        sweep_config, project="adni_multimodal", entity="multimodal_network")
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

        elif config.network == 'multi_loss':
            # get the model
            model = MultiLossModel(
                learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
            # load the data
            data.set_triplet_loss_dataloader()

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

#     wandb.init(project='adni_multimodal', config=config)
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


if __name__ == '__main__':

    # RUN MODELS FOR EVERY SEED

    for seed in seed_list:
        seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(mode=True)

        # main_resnet(seed, config['resnet_config'])
        # main_tabular(seed, config['tabular_config'])
        # main_supervised_multimodal(seed, config['supervised_config'])
        main_daft(seed, config['daft_config'])
        # main_film(seed, config['film_config'])
        # main_triplet(seed, config['triplet_center_config'])

        ########################## TEST ###############################

        # test_resnet(seed, config['resnet_config'])
        # test_tabular(seed, config['tabular_config'])
        # test_supervised_multimodal(seed, config['supervised_config'])
        # test_daft(seed, config['daft_config'])
        # test_film(seed, config['film_config'])
        # test_triplet(seed, config['triplet_center_config'])


##############################################################################
    # run tabular
    # main_tabular(tabular_config)

    # run resnet
    # main_resnet(resnet_config)

    # run multimodal
    # main_supervised_multimodal(supervised_config)

    # run daft
    # main_daft(daft_config)

    # run contrastive learning
    # main_contrastive_learning(contrastive_config)

    # run kfold multimodal (will be tested)
    # main_kfold_multimodal(wandb, wandb_logger, fold_number = 5, learning_rate=1e-3, batch_size=8, max_epochs=100, age=None)

    # run triplet loss model
    # main_triplet(triplet_config)

    # run multiloss model (triplet + cross entropy)
    # main_multiloss(config['supervised_config'])

    # run knn (this can be run after the models for triplet or contrastive loss)
    # knn(knn_config)

    # generate embedding visualizations
    # used the visualize the embeddings gathered from an encoder model (visualized in wandb)
    # wandb.init(project="adni_multimodal"odal', config=knn_config)
    # wandb_logger = WandbLogger()
    # get_embeddings(wandb, wandb_logger)

    # run grid search
    # run_grid_search('multi_loss')
