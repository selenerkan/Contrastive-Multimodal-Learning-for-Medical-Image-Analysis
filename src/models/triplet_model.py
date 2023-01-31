import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
# from monai.networks.nets.resnet_group import resnet10, resnet18, resnet34, resnet50
from models.model_blocks.resnet_block import ResNet
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from pytorch_metric_learning import losses


class TripletModel(LightningModule):
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
        resnet_out_dim = 32
        self.fc2 = nn.Linear(resnet_out_dim + 10, resnet_out_dim + 10)

    def forward(self, img, tab):
        """

        img is the input image data ()
        tab is th einput tabular data

        """
        # run the model for the image
        img = self.resnet(img)
        img = img.view(img.size(0), -1)

        # change the dtype of the tabular data
        tab = tab.to(torch.float32)
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
        img, positive, negative, tab, positive_tab, negative_tab = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5]

        embeddings = self(img, tab)
        pos_embeddings = self(positive, positive_tab)
        neg_embeddings = self(negative, negative_tab)

        loss_function = nn.TripletMarginLoss()
        loss = loss_function(embeddings, pos_embeddings, neg_embeddings)

        # Log loss on every epoch
        self.log('train_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, positive, negative, tab, positive_tab, negative_tab = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5]

        embeddings = self(img, tab)
        pos_embeddings = self(positive, positive_tab)
        neg_embeddings = self(negative, negative_tab)

        loss_function = nn.TripletMarginLoss()
        loss = loss_function(embeddings, pos_embeddings, neg_embeddings)

        # Log loss on every epoch
        self.log('validation_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, positive, negative, tab, positive_tab, negative_tab = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5]

        embeddings = self(img, tab)
        pos_embeddings = self(positive, positive_tab)
        neg_embeddings = self(negative, negative_tab)

        loss_function = nn.TripletMarginLoss()
        loss = loss_function(embeddings, pos_embeddings, neg_embeddings)

        # Log loss on every epoch
        self.log('test_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss
