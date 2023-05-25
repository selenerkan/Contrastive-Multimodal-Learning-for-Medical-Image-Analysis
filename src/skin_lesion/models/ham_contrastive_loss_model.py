import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from models.model_blocks.ham_resnet_block import ResNet
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from pytorch_metric_learning import losses
import torchvision


class HamContrastiveModel(LightningModule):
    '''
    Uses ResNet for the image data, concatenates image and tabular data at the end
    '''

    def __init__(self, learning_rate=0.013, weight_decay=0.01, correlation=False):

        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate
        self.wd = weight_decay
        self.num_classes = 7
        self.feature_dim = 32
        self.embedding_dimension = 64
        self.correlation = correlation

        # IMAGE DATA
        # self.resnet = torchvision.models.resnet18(
        #     weights=torchvision.models.ResNet18_Weights.DEFAULT)  # output features are 1000
        self.resnet = torchvision.models.resnet18()  # output features are 1000
        # change resnet fc output to 128 features
        self.resnet.fc = nn.Linear(512, 128)

        # TABULAR DATA
        # fc layer for tabular data
        self.fc1 = nn.Linear(16, 128)  # tabular data is one-hot-encoded
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)

        # shared FC layer
        self.fc6 = nn.Linear(128, self.embedding_dimension)

        # TABULAR + IMAGE DATA
        # mlp projection head which takes concatenated input
        # arrange the dimension of the projection head according to correlation or concat usage
        if self.correlation:
            concatanation_dimension = (self.embedding_dimension * 2) - 1
        else:
            concatanation_dimension = 128
        # outputs will be used in triplet/center loss
        self.fc7 = nn.Linear(concatanation_dimension, self.feature_dim)

        # MLP head, it will be replaced with an FC layer in classification
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

    def forward(self, img, tab):
        """

        img is the input image data ()
        tab is th einput tabular data

        """
        # reshape image from [2, n_views, 3, 224, 224] to [2*n_views, 3, 224, 224]
        # this adds all views one after the other
        img = img.flatten(0, 1)
        # run the model for the image
        img = self.resnet(img)
        img = self.fc6(F.relu(img))

        # forward pass for tabular data
        tab = tab.to(torch.float32)
        tab = F.relu(self.fc1(tab))
        tab = F.relu(self.fc2(tab))
        tab = F.relu(self.fc3(tab))
        tab = F.relu(self.fc4(tab))
        tab = F.relu(self.fc5(tab))
        tab = self.fc6(tab)
        # double every row of the tabular data to match the size of the images
        tab = torch.repeat_interleave(tab, repeats=2, dim=0)

        # concat image and tabular data
        # arrange the combination process according to the correlation flag
        if self.correlation:
            img = img.unsqueeze(0)
            tab = tab.unsqueeze(1)
            x = F.conv1d(img, tab, padding=self.embedding_dimension -
                         1, groups=img.size(1))
            x = x.squeeze()
        else:
            x = torch.cat((img, tab), dim=1)

        # get the final concatenated embedding
        x = F.relu(self.fc7(x))
        # calculate the output of the mlp head
        out = self.mlp(x)

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
        img, tab, label = batch

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
        img, tab, label = batch

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
        self.log('val_epoch_loss', loss, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):

        # get tabular and image data from the batch
        img, tab, label = batch

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
