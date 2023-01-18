import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
# from monai.networks.nets.resnet import resnet10, resnet18, resnet34, resnet50
from models.model_blocks.resnet_block import ResNet
from torch.nn import Softmax
import torchmetrics
from torch.optim.lr_scheduler import StepLR


class ResNetModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, learning_rate=1e-3, weight_decay=1e-5):

        super().__init__()

        self.lr = learning_rate
        self.wd = weight_decay

        self.softmax = Softmax(dim=1)

        self.resnet = ResNet()

        # add a new fc layer
        resnet_out_dim = self.resnet.fc.out_features
        self.fc = nn.Linear(resnet_out_dim, 3)

        # track accuracy
        self.train_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=3, top_k=1)
        self.val_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=3, top_k=1)

        self.train_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=3, top_k=1)
        self.val_micro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='micro', num_classes=3, top_k=1)

    def forward(self, x):
        """
        x is the image data
        """
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = StepLR(optimizer,
                           step_size=10,  # Period of learning rate decay
                           gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        img, y = batch

        y_pred = self(img)

        loss = F.cross_entropy(y_pred, y.squeeze())

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)

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

        img, y = batch

        y_pred = self(img)

        loss = F.cross_entropy(y_pred, y.squeeze())

        # Log loss
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)

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

        img, y = batch

        y_pred = self(img)

        loss = F.cross_entropy(y_pred, y.squeeze())

        self.log("test_loss", loss)

        return loss
