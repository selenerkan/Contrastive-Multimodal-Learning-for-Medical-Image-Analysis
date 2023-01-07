import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50

import torchmetrics
from pytorch_metric_learning import losses


class ContrastiveModel(LightningModule):
    '''
    Uses ResNet for the image data, concatenates image and tabular data at the end
    '''

    def __init__(self, learning_rate):

        super().__init__()

        self.lr = learning_rate

        # IMAGE DATA
        # output dimension is adapted from simCLR
        self.resnet = resnet10(pretrained=False,
                               spatial_dims=3,
                               n_input_channels=1,
                               )  # output features are 400

        # TABULAR DATA
        # fc layer for tabular data
        self.fc1 = nn.Linear(13, 13)

        # tabular + IMAGE DATA
        # mlp projection head which takes concatenated input
        resnet_out_dim = self.resnet.fc.out_features
        self.mlp = nn.Sequential(
            nn.Linear(resnet_out_dim + 13, resnet_out_dim + 13), nn.ReLU(), nn.Linear(resnet_out_dim + 13, resnet_out_dim + 13))

    def forward(self, img, tab):
        """

        img is the input image data ()
        tab is th einput tabular data

        """
        # reshape the size of images from (3,2,1,120,120,120) to (6,1,120,120,120)
        img = img.flatten(0, 1)
        # run the model for the image
        img = self.resnet(img)

        # change the dtype of the tabular data
        tab = tab.to(torch.float32)
        # concat the tab vectir with the same one to match image dimensions
        tab = torch.cat((tab, tab), dim=0)
        # forward tabular data
        tab = F.relu(self.fc1(tab))

        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)
        out = F.relu(self.mlp(x))
        out = torch.squeeze(out)

        return out

    def configure_optimizers(self):

        # weight decay can be added, lr can be changed
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, tab = batch

        embeddings = self(img, tab)

        # generate same labels for the positive pairs
        # The assumption here is that each image is followed by its positive pair
        # data[0] and data[1], data[2] and data[3] are the positive pairs and so on
        batch_size = img.size(0) * 2
        labels = torch.arange(batch_size)
        labels[1::2] = labels[0::2]

        loss_function = losses.NTXentLoss(
            temperature=0.5)  # temperature value is copied from simCLR
        loss = loss_function(embeddings, labels)

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, tab = batch

        embeddings = self(img, tab)

        # generate same labels for the positive pairs
        # The assumption here is that each image is followed by its positive pair
        # data[0] and data[1], data[2] and data[3] are the positive pairs and so on
        batch_size = img.size(0) * 2
        labels = torch.arange(batch_size)
        labels[1::2] = labels[0::2]

        loss_function = losses.NTXentLoss(
            temperature=0.5)  # temperature value is copied from simCLR
        loss = loss_function(embeddings, labels)

        # Log loss on every epoch
        self.log('validation_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, tab = batch

        embeddings = self(img, tab)

        # generate same labels for the positive pairs
        # The assumption here is that each image is followed by its positive pair
        # data[0] and data[1], data[2] and data[3] are the positive pairs and so on
        batch_size = img.size(0) * 2
        labels = torch.arange(batch_size)
        labels[1::2] = labels[0::2]

        loss_function = losses.NTXentLoss(
            temperature=0.5)  # temperature value is copied from simCLR
        loss = loss_function(embeddings, labels)

        # Log loss on every epoch
        self.log('test_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss
