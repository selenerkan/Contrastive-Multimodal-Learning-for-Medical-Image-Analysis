import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets.resnet_group import resnet10, resnet18, resnet34, resnet50
from torch.nn import Softmax


class ResNetModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, learning_rate, weight_decay):

        super().__init__()

        self.lr = learning_rate
        self.wd = weight_decay

        self.softmax = Softmax(dim=1)

        self.resnet = resnet10(pretrained=False,
                               spatial_dims=3,
                               n_input_channels=1,
                               )

        # add a new fc layer
        self.fc1 = nn.Linear(400, 400)
        self.fc2 = nn.Linear(400, 3)

        # combine the nets
        self.net = nn.Sequential(self.resnet, self.fc1, self.fc2)

        # track accuracy
        self.train_acc = []
        self.val_acc = []

    def forward(self, x):
        """

        x is the input data

        """
        out = self.net(x)
        out = torch.squeeze(F.relu(out))

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)

        return optimizer

    def training_step(self, batch, batch_idx):

        img, y = batch

        y_pred = self(img)

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

        for i in range(len(pred_label)):
            if pred_label[i] == y[i]:
                self.train_acc.append(1)
            else:
                self.train_acc.append(0)

        return loss

    def training_epoch_end(self, outputs):
        acc = sum(self.train_acc) / len(self.train_acc)
        self.log('train_epoch_acc', acc, on_epoch=True, on_step=False)
        self.train_acc = []

    def validation_step(self, batch, batch_idx):

        img, y = batch

        y_pred = self(img)

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

        for i in range(len(pred_label)):
            if pred_label[i] == y[i]:
                self.val_acc.append(1)
            else:
                self.val_acc.append(0)

        return loss

    def validation_epoch_end(self, outputs):
        acc = sum(self.val_acc) / len(self.val_acc)
        self.log('val_epoch_acc', acc, on_epoch=True, on_step=False)
        self.val_acc = []

    def test_step(self, batch, batch_idx):

        img, y = batch

        y_pred = self(img)

        loss = F.cross_entropy(y_pred, y.squeeze())

        self.log("test_loss", loss)

        return loss
