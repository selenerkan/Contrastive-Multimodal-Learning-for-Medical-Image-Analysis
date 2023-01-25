import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
# from monai.networks.nets.resnet import resnet10, resnet18, resnet34, resnet50
from models.model_blocks.resnet_block import ResNet
import torchmetrics
from torch.nn import Softmax
from torch.optim.lr_scheduler import StepLR


class TabularModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, learning_rate, weight_decay=1e-5):

        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate
        self.wd = weight_decay

        # TABULAR
        # fc layer for tabular data
        self.fc1 = nn.Linear(13, 13)
        self.fc2 = nn.Linear(13, 13)
        self.fc3 = nn.Linear(13, 13)
        self.fc4 = nn.Linear(13, 10)
        self.fc5 = nn.Linear(10, 3)

        # track accuracy
        self.train_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=3, top_k=1)
        self.val_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=3, top_k=1)

        self.train_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=3, top_k=1)
        self.val_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=3, top_k=1)

        self.softmax = Softmax(dim=1)

    def forward(self, tab):
        """
        img is the input image data
        tab is th einput tabular data
        """

        # forward pass for tabular data
        tab = tab.to(torch.float32)
        tab = F.relu(self.fc1(tab))
        tab = F.relu(self.fc2(tab))
        tab = F.relu(self.fc3(tab))
        tab = F.relu(self.fc4(tab))
        out = self.fc5(tab)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)

        return optimizer

    def training_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(tab)

        loss = F.cross_entropy(y_pred, y.squeeze())

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)

        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # calculate and log accuracy
        train_acc = self.train_macro_accuracy(pred_label, y)
        self.log('train_macro_acc', train_acc, on_epoch=True, on_step=False)

        # calculate and log accuracy
        train_micro_acc = self.train_micro_accuracy(pred_label, y)
        self.log('train_micro_acc', train_micro_acc,
                 on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(tab)

        loss = F.cross_entropy(y_pred, y.squeeze())

        # Log loss
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)

        # calculate acc
        # take softmax
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
        y_pred_softmax = self.softmax(y_pred)

        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        # calculate and log accuracy
        val_acc = self.val_macro_accuracy(pred_label, y)
        self.log('val_macro_acc', val_acc, on_epoch=True, on_step=False)

        # calculate and log accuracy
        val_micro_acc = self.val_micro_accuracy(pred_label, y)
        self.log('val_micro_acc', val_micro_acc, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(tab)

        loss = F.cross_entropy(y_pred, y.squeeze())

        self.log("test_loss", loss)

        return loss
