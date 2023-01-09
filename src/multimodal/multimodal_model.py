import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
import torchmetrics


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

        # self.train_acc = torchmetrics.Accuracy()
        # self.valid_acc = torchmetrics.Accuracy()

        self.metrics = {"train_epoch_losses": [], "train_accuracy": [],
                        "val_epoch_losses": [], "valid_accuracy": []}

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
        # y = y_int.to(torch.float32)

        y_pred = self(img, tab)

        # loss = F.binary_cross_entropy(y_pred, y.squeeze())
        loss = F.cross_entropy(y_pred, y.squeeze())

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)
        self.metrics["train_epoch_losses"].append(loss.detach())

        # Calculate accuracy
        # self.train_acc(y_pred.unsqueeze(0), y_int)

    # def on_train_epoch_end(self):
    #     # Log accuracy
    #     train_acc = self.train_acc.compute()
    #     # self.log("train_epochwise_accuracy", train_acc)
    #     self.metrics["train_accuracy"].append(train_acc)

    def validation_step(self, batch, batch_idx):

        img, tab, y = batch
        # y = y_int.to(torch.float32)

        y_pred = self(img, tab)

        loss = F.cross_entropy(y_pred, y.squeeze())

        # Log loss
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)
        self.metrics["val_epoch_losses"].append(loss.detach())

        # # Calculate accuracy
        # self.valid_acc(y_pred.unsqueeze(0), y_int)
        # # self.metrics["valid_accuracy"].append(loss)

        # return loss

    # def on_validation_end(self):
    #     # Log accuracy
    #     valid_acc = self.valid_acc.compute()
    #     # self.log("valid_epochwise_accuracy", valid_acc)
    #     self.metrics["valid_accuracy"].append(valid_acc)

    def test_step(self, batch, batch_idx):

        img, tab, y = batch
        # y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss = F.cross_entropy(y_pred, y.squeeze())

        self.log("test_loss", loss)

        return loss
