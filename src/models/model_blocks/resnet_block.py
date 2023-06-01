from pytorch_lightning.core.module import LightningModule
import torch.nn as nn


def conv3d(in_channels, out_channels, kernel_size=3, stride=1):
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)


class ConvBnReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        # self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        # out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, stride=1):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        # self.bn1 = nn.BatchNorm3d(out_channels, momentum=1.0)
        # self.bn1 = nn.GroupNorm(4, out_channels)
        self.conv2 = conv3d(out_channels, out_channels)
        # self.bn2 = nn.BatchNorm3d(out_channels, momentum=1.0)
        # self.bn2 = nn.GroupNorm(4, out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels,
                       kernel_size=1, stride=stride),
                # nn.BatchNorm3d(out_channels, momentum=1.0),
                # nn.GroupNorm(4, out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(LightningModule):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4) -> None:
        super().__init__()

        self.conv1 = ConvBnReLU(
            in_channels, n_basefilters, bn_momentum=bn_momentum)
        # self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(
            n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(
            n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(
            2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(
            4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    def forward(self, image):
        out = self.conv1(image)
        # out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
