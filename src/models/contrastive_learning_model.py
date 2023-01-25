import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
# from monai.networks.nets.resnet_group import resnet10, resnet18, resnet34, resnet50
from models.model_blocks.resnet_block import ResNet
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from pytorch_metric_learning import losses


class ContrastiveModel(LightningModule):
    '''
    Uses ResNet for the image data, concatenates image and tabular data at the end
    '''

    def __init__(self, learning_rate=0.013, weight_decay=0.01):

        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate
        self.wd = weight_decay

        # IMAGE DATA
        # output dimension is adapted from simCLR
        self.resnet = ResNet()  # output features are 32

        # TABULAR DATA
        # fc layer for tabular data
        self.fc1 = nn.Linear(13, 10)

        # TABULAR + IMAGE DATA
        # mlp projection head which takes concatenated input
        # self.mlp = nn.Sequential(
        #     nn.Linear(resnet_out_dim + 13, resnet_out_dim + 13), nn.ReLU(), nn.Linear(resnet_out_dim + 13, resnet_out_dim + 13))
        resnet_out_dim = 32
        self.fc2 = nn.Linear(resnet_out_dim + 10, resnet_out_dim + 10)

    def forward(self, img, tab):
        """

        img is the input image data ()
        tab is th einput tabular data

        """
        # reshape the size of images from (3,2,1,120,120,120) to (6,1,120,120,120)
        img = img.flatten(0, 1)
        # run the model for the image
        img = self.resnet(img)
        img = img.view(img.size(0), -1)

        # change the dtype of the tabular data
        tab = tab.to(torch.float32)
        # concat the tab vectir with the same one to match image dimensions
        tab = torch.cat((tab, tab), dim=0)
        # forward tabular data
        tab = F.relu(self.fc1(tab))

        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)
        out = self.fc2(x)

        return out

    def configure_optimizers(self):

        # weight decay can be added, lr can be changed
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler = MultiStepLR(optimizer,
        #                         # List of epoch indices
        #                         milestones=[18, 27],
        #                         gamma=0.1)  # Multiplicative factor of learning rate decay

        # return [optimizer], [scheduler]
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
