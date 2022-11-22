import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F


class AdniModel(LightningModule):

    def __init__(self):

        super().__init__()

        """

    The convolutions are arranged in such a way that the image maintain the x and y dimensions. only the channels change

    """

        self.layer_1 = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(
            3, 3, 3))

        # self.layer_2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(
        #     3, 3), padding=(1, 1), stride=(1, 1))

        # self.layer_3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(
        #     3, 3), padding=(1, 1), stride=(1, 1))

        # self.pool = nn.MaxPool2d(kernel_size=(
        #     3, 3), padding=(1, 1), stride=(1, 1))

        # # the input dimensions are (Number of dimensions * height * width)
        # self.layer_5 = nn.Linear(12*50*50, 1000)

        # self.layer_6 = nn.Linear(1000, 100)

        # self.layer_7 = nn.Linear(100, 50)

        # self.layer_8 = nn.Linear(50, 10)

        # self.layer_9 = nn.Linear(10, 5)

    def forward(self, x):
        """

        x is the input data

        """
        x = torch.unsqueeze(x, 0)

        x = self.layer_1(x)

        print(x.size())

        return torch.tensor(1.0, dtype=torch.float).unsqueeze(0)

        # x = self.pool(x)

        # x = self.layer_2(x)

        # x = self.pool(x)

        # x = self.layer_3(x)

        # x = self.pool(x)

        # x = x.view(x.size(0), -1)

        # print(x.size())

        # x = self.layer_5(x)

        # x = self.layer_6(x)

        # x = self.layer_7(x)

        # x = self.layer_8(x)

        # x = self.layer_9(x)

        return x

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-7)

        return optimizer

    """

The Pytorch-Lightning module handles all the iterations of the epoch

"""

    def training_step(self, batch, batch_idx):

        x, y = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred, y)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred, y)

        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred, y)

        self.log("loss", loss)

        return loss
