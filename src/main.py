import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

from adni_dataset import AdniDataModule, KfoldMultimodalDataModule

from conv3D.model import AdniModel
from models.resnet_model import ResNetModel
from models.multimodal_model import MultiModModel
from models.tabular_model import TabularModel
from models.contrastive_learning_model import ContrastiveModel
from models.triplet_model import TripletModel

from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from settings import CSV_FILE, SEED, CHECKPOINT_DIR, resnet_config, supervised_config, contrastive_config, tabular_config, triplet_config, knn_config
import torch.multiprocessing
from datetime import datetime
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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


def main_tabular(config=None):
    '''
    main function to run the multimodal architecture
    '''
    wandb.init(project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = TabularModel(learning_rate=wandb.config.learning_rate,
                         weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(
        CSV_FILE, age=wandb.config.age, batch_size=wandb.config.batch_size, spatial_size=wandb.config.spatial_size)
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'tabular'), filename='lr='+str(wandb.config.learning_rate)+'_wd='+str(wandb.config.weight_decay)+'_'+dt_string+'-{epoch:03d}')

    # Add learning rate scheduler monitoring
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def main_resnet(config=None):
    '''
    main function to run the resnet architecture
    '''
    print('YOU ARE RUNNING RESNET')
    print(config)

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
    data.prepare_data()
    data.set_resnet_dataset()
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'resnet'), filename='lr='+str(wandb.config.learning_rate)+'_wd='+str(wandb.config.weight_decay)+'_'+dt_string+'-{epoch:03d}')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def main_supervised_multimodal(config=None):
    '''
    main function to run the supervised multimodal architecture
    '''

    print('YOU ARE RUNNING SUPERVISED MULTIMODAL')
    print(config)

    wandb.init(project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = MultiModModel(learning_rate=wandb.config.learning_rate,
                          weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # check if the checkpoint flag is True
    if wandb.config.checkpoint_flag:
        # copy the weights from multimodal supervised model checkpoint
        model = MultiModModel.load_from_checkpoint(
            config.checkpoint, learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    elif wandb.config.contrastive_checkpoint_flag:
        contrastive_model = ContrastiveModel.load_from_checkpoint(
            wandb.config.contrastive_checkpoint)
        # copy the resnet and fc1 weights from contrastive learning model
        model.resnet = contrastive_model.resnet
        model.fc1 = contrastive_model.fc1

        # freeze network weights
        # model.resnet.freeze()
        # model.fc1.requires_grad_(False)

    # load the data
    data = AdniDataModule(
        CSV_FILE, age=wandb.config.age, batch_size=wandb.config.batch_size, spatial_size=wandb.config.spatial_size)
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'supervised'), filename='lr='+str(wandb.config.learning_rate)+'_wd='+str(wandb.config.weight_decay)+'_'+dt_string+'-{epoch:03d}')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


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

    print('YOU ARE RUNNING CONTRASTIVE MODEL')
    print(config)

    wandb.init(project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = ContrastiveModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(
        CSV_FILE, age=wandb.config.age, batch_size=wandb.config.batch_size, spatial_size=wandb.config.spatial_size)
    data.prepare_data()
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'contrastive'), filename='lr='+str(wandb.config.learning_rate)+'_wd='+str(wandb.config.weight_decay)+'_'+dt_string+'-{epoch:03d}')

    # Add learning rate scheduler monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=wandb.config.max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor], log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def main_triplet(config=None):
    '''
    main function to run the multimodal architecture
    '''

    print('YOU ARE RUNNING TRIPLET LOSS MODEL')
    print(config)

    wandb.init(project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # get the model
    model = TripletModel(
        learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    wandb.watch(model, log="all")

    # load the data
    data = AdniDataModule(
        CSV_FILE, age=wandb.config.age, batch_size=wandb.config.batch_size, spatial_size=wandb.config.spatial_size)
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
        dirpath=os.path.join(CHECKPOINT_DIR, 'triplet'), filename='lr='+str(wandb.config.learning_rate)+'_wd='+str(wandb.config.weight_decay)+'_'+dt_string+'-{epoch:03d}')

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
            'batch_size': {'value': 32},
            'max_epochs': {'value': 30},
            'age': {'value': None},
            'spatial_size': {'value': (120, 120, 120)},
            'learning_rate': {'values': [0.03, 0.013, 0.0055, 0.0023, 0.001]},
            'weight_decay': {'values': [0, 1e-2, 1e-4]},
            # checkpoints are implemented in case grid search wants to be tried with checkpoint weights
            # put the checkpoint path here
            'checkpoint': {'value': r'/home/guests/selen_erkan/experiments/checkpoints/supervised/25.01.2023-18.49-epoch=029.ckpt'},
            # this is only for using contrastive model weights in supervised model
            'contrastive_checkpoint': {'value': r'/home/guests/selen_erkan/experiments/checkpoints/contrastive/25.01.2023-17.14-epoch=029.ckpt'},
            'checkpoint_flag': {'value': False},
            'contrastive_checkpoint_flag': {'value': False}
        }
    }

    count = len(sweep_config['parameters']['learning_rate']['values']) * \
        len(sweep_config['parameters']['weight_decay']['values'])

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

        # load the data
        data = AdniDataModule(
            CSV_FILE, age=config.age, batch_size=config.batch_size, spatial_size=config.spatial_size)
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

            # below is used if grid search wants to be tried with checkpoint weights
            # check if the checkpoint flag is True
            if config.checkpoint_flag:
                # copy the weights from multimodal supervised model checkpoint
                model = MultiModModel.load_from_checkpoint(
                    config.checkpoint, learning_rate=config.learning_rate, weight_decay=config.weight_decay)

            elif config.contrastive_checkpoint_flag:
                contrastive_model = ContrastiveModel.load_from_checkpoint(
                    config.contrastive_checkpoint)

                # copy the resnet and fc1 weights from contrastive learning model
                model.resnet = contrastive_model.resnet
                model.fc1 = contrastive_model.fc1

            data.set_supervised_multimodal_dataloader()

        elif config.network == 'contrastive':
            # get the model
            model = ContrastiveModel(learning_rate=config.learning_rate,
                                     weight_decay=config.weight_decay)
            # load the data
            data.set_contrastive_loss_dataloader()

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


def knn():
    pass


def run_knn(config):

    print('YOU ARE RUNNING KNN FOR', config['model'])
    print(config)

    wandb.init(project="multimodal_training",
               entity="multimodal_network", config=config)
    wandb_logger = WandbLogger()

    # copy the weights from the checkpoint
    # use triplet model for both triplet loss and contrastive loss because the models are exactly the same
    model = TripletModel.load_from_checkpoint(
        config.checkpoint, learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

    # set the model to eval mode to run knn
    torch.set_grad_enabled(False)
    model.eval()

    # load the data
    data = AdniDataModule(
        CSV_FILE, age=wandb.config.age, batch_size=wandb.config.batch_size, spatial_size=wandb.config.spatial_size)
    data.prepare_data()
    # use supervised multimodal dataset because we only need images and tabular data woth labels (no negative and positive pairs for contrastive learning)
    data.set_supervised_multimodal_dataloader()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    train_encodings = []
    val_encodings = []
    train_labels = []
    val_labels = []

    # get the training batches and calculate the encodings
    for step, batch in enumerate(train_dataloader):

        img, tab, label = batch[0], batch[3], batch[-1]
        # calculate the encodings
        encodings = model(img, tab)
        # add encodings to the list
        train_encodings.extend(list(encodings))
        # add the labels to the list
        train_labels.extend(list(label))

    # get the validation batches and calculate the encodings
    for step, batch in enumerate(val_dataloader):

        img, tab, label = batch[0], batch[3], batch[-1]
        # calculate the encodings
        encodings = model(img, tab)
        # add encodings to the list
        val_encodings.extend(list(encodings))
        # add the labels to the list
        val_labels.extend(list(label))

    # run knn model

    pass


if __name__ == '__main__':

    # set the seed of the environment
    # Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    seed_everything(SEED, workers=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # run tabular
    # main_tabular(tabular_config)

    # run resnet
    # main_resnet(resnet_config)

    # run multimodal
    # main_supervised_multimodal(supervised_config)

    # run contrastive learning
    # main_contrastive_learning(contrastive_config)

    # run kfold multimodal
    # main_kfold_multimodal(wandb, wandb_logger, fold_number = 5, learning_rate=1e-3, batch_size=8, max_epochs=100, age=None)

    # run triplet loss model
    main_triplet(triplet_config)

    # run grid search
    # run_grid_search('contrastive')
