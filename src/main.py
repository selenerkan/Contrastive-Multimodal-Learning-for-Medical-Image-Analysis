import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from conv3D.model import AdniModel
from dataset import AdniDataModule
from multimodal_dataset import MultimodalDataModule, KfoldMultimodalDataModule
from contrastive_loss_dataset import ContrastiveDataModule

from ResNet.model import ResNetModel
from multimodal.multimodal_model import MultiModModel
from multimodal.contrastive_learning_model import ContrastiveModel

from settings import CSV_FILE


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
    trainer = Trainer(max_epochs=15, logger=wandb_logger)
    trainer.fit(model, data)


def main_resnet(wandb, wandb_logger):
    '''
    main function to run the resnet architecture
    '''
    # ge the model
    model = ResNetModel()

    # load the data
    data = AdniDataModule(CSV_FILE)

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(max_epochs=15, logger=wandb_logger)
    trainer.fit(model, data)


def main_multimodal(wandb, wandb_logger, learning_rate=1e-3, batch_size=8, max_epochs=60, age=None):
    '''
    main function to run the multimodal architecture
    '''
    # get the model
    model = MultiModModel(learning_rate=learning_rate)

    csv_dir = CSV_FILE

    # load the data
    data = MultimodalDataModule(csv_dir, age=age, batch_size=batch_size)

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(accelerator="gpu", devices=1,
                      max_epochs=max_epochs, logger=wandb_logger)
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

    fold_num = 0
    # train the mdoel
    for train_dataloader, val_dataloader in zip(train_dataloaders, val_dataloaders):
        # get the model
        model = MultiModModel(learning_rate=learning_rate)
        trainer = Trainer(accelerator="gpu", devices=1,
                          max_epochs=max_epochs, logger=wandb_logger)
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


def main_contrastive_learning(wandb, wandb_logger, learning_rate=1e-3, batch_size=8, max_epochs=60, age=None):
    '''
    main function to run the multimodal architecture
    '''
    # get the model
    model = ContrastiveModel(learning_rate=learning_rate)

    csv_dir = CSV_FILE

    # load the data
    data = ContrastiveDataModule(csv_dir, age=age, batch_size=batch_size)

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(accelerator="gpu", devices=1,
                      max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model, data)


if __name__ == '__main__':

    # create wandb objects to track runs
    # wandb.init(project="multimodal-network-test")

    wandb.init(project="multimodal_training", entity="multimodal_network")
    wandb_logger = WandbLogger()

    # # run conv3d
    # main_conv3d(wandb, wandb_logger)

    # run resnet
    # main_resnet(wandb, wandb_logger)

    # run multimodal
    main_multimodal(wandb, wandb_logger, learning_rate=1e-3,
                    batch_size=8, max_epochs=60, age=None)

    # run kfold multimodal
    # main_kfold_multimodal(wandb, wandb_logger, fold_number = 5, learning_rate=1e-3, batch_size=8, max_epochs=60, age=None)

    # run contrastive learning
    # main_contrastive_learning(wandb, wandb_logger, learning_rate=1e-3, batch_size=8, max_epochs=60, age=None)
