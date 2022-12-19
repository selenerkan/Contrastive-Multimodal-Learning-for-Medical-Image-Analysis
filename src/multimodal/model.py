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

    def __init__(self):

        super().__init__()

        self.resnet = resnet10(pretrained=False,
                               spatial_dims=3,
                               n_input_channels=1,
                               )

        # fc layer to make image size same as tabular data size
        # self.fc = nn.Linear(400, 1)

        # combine resnet with final fc layer
        # self.imagenet = nn.Sequential(self.resnet, self.fc)

        # fc layer for tabular data
        self.fc1 = nn.Linear(13, 13)

        # first fc layer which takes concatenated imput
        self.fc2 = nn.Linear(413, 200)

        # final fc layer which takes concatenated imput
        self.fc3 = nn.Linear(200, 1)

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

        self.metrics = {"train_epoch_losses": [], "train_accuracy": [],
                        "val_epoch_losses": [], "valid_accuracy": []}

    def forward(self, img, tab):
        """

        img is the input image data
        tab is th einput tabular data

        """

        # run the model for the image
        img = torch.unsqueeze(img, 1)
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

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer

    def training_step(self, batch, batch_idx):

        img, tab, y_int = batch
        y = y_int.to(torch.float32)

        y_pred = torch.sigmoid(self(img, tab))

        loss = F.binary_cross_entropy(y_pred, y.squeeze())

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)
        self.metrics["train_epoch_losses"].append(loss)


    def validation_step(self, batch, batch_idx):

        img, tab, y_int = batch
        y = y_int.to(torch.float32)

        y_pred = self(img, tab)

        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze())

        # Log loss
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)
        self.metrics["val_epoch_losses"].append(loss)


    def test_step(self, batch, batch_idx):

        img, tab, y = batch
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze())

        self.log("loss", loss)

        return loss
