import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
# from monai.networks.nets.resnet import resnet10, resnet18, resnet34, resnet50
from models.model_blocks.resnet_block import ResNet
import torchmetrics
from torch.nn import Softmax
from torch.optim.lr_scheduler import StepLR


class MultiModModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, learning_rate, weight_decay=1e-5):

        super().__init__()
        self.lr = learning_rate
        self.wd = weight_decay

        # IMAGE
        # resnet module for image data
        self.resnet = ResNet(in_channels=1, n_outputs=3,
                             bn_momentum=0.1, n_basefilters=64)

        # TABULAR
        # fc layer for tabular data
        self.fc1 = nn.Linear(13, 13)
        self.fc2 = nn.Linear(13, 13)

        # TABULAR + IMAGE DATA
        # mlp projection head which takes concatenated input
        resnet_out_dim = self.resnet.fc.out_features
        self.mlp = nn.Sequential(
            nn.Linear(resnet_out_dim + 13, resnet_out_dim + 13), nn.ReLU(), nn.Linear(resnet_out_dim + 13, 3))

        # track accuracy
        self.train_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=3, top_k=1)
        self.val_macro_accuracy = torchmetrics.Accuracy(
            task='multiclass', average='macro', num_classes=3, top_k=1)

        self.softmax = Softmax(dim=1)

    def forward(self, img, tab):
        """
        img is the input image data
        tab is th einput tabular data
        """

        # run the model for the image
        img = self.resnet(img)

        # change the dtype of the tabular data
        tab = tab.to(torch.float32)
        # forward tabular data
        tab = self.fc2(F.relu(self.fc1(tab)))

        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)
        out = self.mlp(x)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = StepLR(optimizer,
                           step_size=20,  # Period of learning rate decay
                           gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(img, tab)

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

        return loss

    def validation_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(img, tab)

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

        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(img, tab)

        loss = F.cross_entropy(y_pred, y.squeeze())

        self.log("test_loss", loss)

        return loss
