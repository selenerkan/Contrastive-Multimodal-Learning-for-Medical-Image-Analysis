import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F


class AdniModel(LightningModule):

    def __init__(self):

        super().__init__()

        """

    The convolutions are arranged in such a way that the image maintain the x and y dimensions. only the channels change

    """

        self.layer_1 = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(
            3, 3, 3), stride=(1, 1, 1))  # input (192x192x160) output (3x158x190x190)

        self.pool1 = nn.MaxPool3d(kernel_size=(
            2, 2, 2), stride=(2, 2, 2))  # input (3x158x190x190) output (3x79x95x95)

        self.layer_2 = nn.Conv3d(in_channels=3, out_channels=6, kernel_size=(
            3, 3, 3), stride=(1, 1, 1))  # input (3x79x95x95) output (6x77x93x93)

        self.pool2 = nn.MaxPool3d(kernel_size=(
            3, 3, 3), stride=(2, 2, 2))  # input (6x77x93x93) output (6x39x47x47)

        self.layer_3 = nn.Conv3d(in_channels=6, out_channels=12, kernel_size=(
            3, 3, 3), stride=(1, 1, 1))  # input (6x39x47x47) output (12x37x45x45)

        self.pool3 = nn.MaxPool3d(kernel_size=(
            3, 3, 3), stride=(2, 2, 2))  # input (12x37x45x45) output (12x19x23x23)

        # # the input dimensions are (Number of dimensions * height * width)
        self.layer_4 = nn.Linear(181656, 1000)

        self.layer_5 = nn.Linear(1000, 100)

        self.layer_6 = nn.Linear(100, 3)

    def forward(self, x):
        """

        x is the input data

        """
        x = torch.unsqueeze(x, 0)

        x = self.pool1(F.relu(self.layer_1(x)))
        x = self.pool2(F.relu(self.layer_2(x)))
        x = self.pool3(F.relu(self.layer_3(x)))

        x = x.view(x.size(0), -1)
        print('flattened size: ', x.size())

        x = F.relu(self.layer_4(x))

        x = F.relu(self.layer_5(x))

        out = self.layer_6(x)

        out = torch.squeeze(out)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-7)

        return optimizer

    """

The Pytorch-Lightning module handles all the iterations of the epoch

"""

    def training_step(self, batch, batch_idx):

        x, y = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred, y.squeeze())

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred, y.squeeze())

        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred, y.squeeze())

        self.log("loss", loss)

        return loss
