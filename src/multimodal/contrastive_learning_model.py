import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
# from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
import torchvision.models as models

import torchmetrics
from pytorch_metric_learning import losses


class ContrastiveModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self):

        super().__init__()

        # IMAGE DATA
        # output dimension is adapted from simCLR
        self.resnet = models.resnet50(pretrained=False, num_classes=2048)

        # TABULAR DATA
        # fc layer for tabular data
        self.fc1 = nn.Linear(13, 13)

        # tabular + IMAGE DATA
        # mlp projection head which takes concatenated input
        resnet_out_dim = self.resnet.fc.out_features
        self.mlp = nn.Sequential(
            nn.Linear(resnet_out_dim + 13, resnet_out_dim + 13), nn.ReLU(), nn.Linear(resnet_out_dim + 13, resnet_out_dim + 13))

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
        # img = torch.unsqueeze(img, 1)
        img = self.resnet(img)

        # change the dtype of the tabular data
        tab = tab.to(torch.float32)
        # forward tabular data
        tab = F.relu(self.fc1(tab))

        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)
        out = F.relu(self.mlp(x))
        out = torch.squeeze(out)

        return out

    def configure_optimizers(self):

        # weight decay can be added, lr can be changed
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer

    def training_step(self, batch, batch_idx):

        img, tab = batch

        # concat the tab vectir with the same one to match image dimensions
        tab = torch.cat((tab, tab), dim=0)
        embeddings = torch.sigmoid(self(img, tab))

        # concat the images list
        img = torch.cat(img, dim=0)  # output=(8,1,120,120,120)

        # generate same labels for the positive pairs
        # The assumption here is that data[0] and data[0+batch_size] are a positive pair
        # Given batch_size=4 data[0] and data[4], data[1] and data[5] are the positive pairs
        batch_size = img.size(0)
        labels = torch.arange(batch_size/2)
        labels = torch.cat((labels, labels), dim=0)

        loss_function = losses.NTXentLoss(
            temperature=0.5)  # temperature value is copied from simCLR
        loss = loss_function(embeddings, labels)

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)
        self.metrics["train_epoch_losses"].append(loss)

        return loss

    def validation_step(self, batch, batch_idx):

        img, tab = batch

        # concat the tab vectir with the same one to match image dimensions
        tab = torch.cat((tab, tab), dim=0)
        embeddings = torch.sigmoid(self(img, tab))

        # concat the images list
        img = torch.cat(img, dim=0)  # output=(8,1,120,120,120)

        # generate same labels for the positive pairs
        # The assumption here is that data[0] and data[0+batch_size] are a positive pair
        # Given batch_size=4 data[0] and data[4], data[1] and data[5] are the positive pairs
        batch_size = img.size(0)
        labels = torch.arange(batch_size/2)
        labels = torch.cat((labels, labels), dim=0)

        loss_function = losses.NTXentLoss(
            temperature=0.5)  # temperature value is copied from simCLR
        loss = loss_function(embeddings, labels)

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)
        self.metrics["train_epoch_losses"].append(loss)

        return loss

    def test_step(self, batch, batch_idx):

        img, tab = batch

        # concat the tab vectir with the same one to match image dimensions
        tab = torch.cat((tab, tab), dim=0)
        embeddings = torch.sigmoid(self(img, tab))

        # concat the images list
        img = torch.cat(img, dim=0)  # output=(8,1,120,120,120)

        # generate same labels for the positive pairs
        # The assumption here is that data[0] and data[0+batch_size] are a positive pair
        # Given batch_size=4 data[0] and data[4], data[1] and data[5] are the positive pairs
        batch_size = img.size(0)
        labels = torch.arange(batch_size/2)
        labels = torch.cat((labels, labels), dim=0)

        loss_function = losses.NTXentLoss(
            temperature=0.5)  # temperature value is copied from simCLR
        loss = loss_function(embeddings, labels)

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)
        self.metrics["train_epoch_losses"].append(loss)

        return loss
