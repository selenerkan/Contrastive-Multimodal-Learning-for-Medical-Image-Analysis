import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
import torchmetrics
from torch.nn import Softmax


class MultiModModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, learning_rate):

        super().__init__()

        self.lr = learning_rate

        # resnet module for image data
        self.resnet = resnet10(pretrained=False,
                               spatial_dims=3,
                               n_input_channels=1,
                               )

        # fc layer for tabular data
        self.fc1 = nn.Linear(13, 13)

        # first fc layer which takes concatenated imput
        self.fc2 = nn.Linear(413, 413)

        # final fc layer which takes concatenated imput
        self.fc3 = nn.Linear(413, 3)

        # track accuracy
        # self.train_acc = torchmetrics.Accuracy()
        # self.valid_acc = torchmetrics.Accuracy()
        self.train_acc = []
        self.val_acc = []

        self.softmax = Softmax()

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
        tab = F.relu(self.fc1(tab))

        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        out = torch.squeeze(out)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(img, tab)

        loss = F.cross_entropy(y_pred, y.squeeze())

        # log step metric
        # self.train_acc(y_pred.unsqueeze(0), y)

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)
        # self.log('train_epoch_acc', self.train_acc,
        #          on_step=False, on_epoch=True)

        # calculate acc
        # take softmax
        y_pred_softmax = self.softmax(y_pred)
        # get the index of max value
        pred_label = torch.argmax(y_pred_softmax, dim=1)

        for i in range(len(pred_label)):
            if pred_label[i] == y[i]:
                self.train_acc.append(1)
            else:
                self.train_acc.append(0)

        acc = sum(self.train_acc) / len(self.train_acc)
        self.log('train_acc_loss', acc, on_epoch=True, on_step=False)

        return loss

    def training_epoch_end(self, outputs):
        self.train_acc = []

    def validation_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(img, tab)

        loss = F.cross_entropy(y_pred, y.squeeze())

        # calculate acc
        # self.valid_acc(y_pred.unsqueeze(0), y)

        # Log loss
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)
        # self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y = batch

        y_pred = self(img, tab)

        loss = F.cross_entropy(y_pred, y.squeeze())

        self.log("test_loss", loss)

        return loss
