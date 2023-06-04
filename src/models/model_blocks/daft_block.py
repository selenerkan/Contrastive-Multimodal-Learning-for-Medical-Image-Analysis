from collections import OrderedDict
import torch
from typing import Any, Dict, Optional, Sequence
from abc import ABCMeta, abstractmethod
from torch.nn import Module
from models.model_blocks.resnet_block import conv3d, ConvBnReLU, ResBlock
import torch.nn as nn


def check_is_unique(values: Sequence[Any]) -> bool:
    if len(values) != len(set(values)):
        raise ValueError("values of list must be unique")


class BaseModel(Module, metaclass=ABCMeta):
    """Abstract base class for models that can be executed by
    :class:`daft.training.train_and_eval.ModelRunner`.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        check_is_unique(self.input_names)
        check_is_unique(self.output_names)
        check_is_unique(list(self.input_names) + list(self.output_names))

    @property
    @abstractmethod
    def input_names(self) -> Sequence[str]:
        """Names of parameters passed to self.forward"""

    @property
    @abstractmethod
    def output_names(self) -> Sequence[str]:
        """Names of tensors returned by self.forward"""


class FilmBase(nn.Module, metaclass=ABCMeta):
    """Absract base class for models that are related to FiLM of Perez et al"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float,
        stride: int,
        ndim_non_img: int,
        location: int,
        activation: str,
        scale: bool,
        shift: bool,
    ) -> None:

        super().__init__()

        # sanity checks
        if location not in set(range(5)):
            raise ValueError(f"Invalid location specified: {location}")
        if activation not in {"tanh", "sigmoid", "linear"}:
            raise ValueError(f"Invalid location specified: {location}")
        if (not isinstance(scale, bool) or not isinstance(shift, bool)) or (not scale and not shift):
            raise ValueError(
                f"scale and shift must be of type bool:\n    -> scale value: {scale}, "
                "scale type {type(scale)}\n    -> shift value: {shift}, shift type: {type(shift)}"
            )
        # ResBlock
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(
            out_channels, momentum=bn_momentum, affine=(location != 3))
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels,
                       kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None
        # Film-specific variables
        self.location = location
        if self.location == 2 and self.downsample is None:
            raise ValueError(
                "This is equivalent to location=1 and no downsampling!")
        # location decoding
        self.film_dims = 0
        if location in {0, 1, 2}:
            self.film_dims = in_channels
        elif location in {3, 4}:
            self.film_dims = out_channels
        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        elif activation == "linear":
            self.scale_activation = None

    @abstractmethod
    def rescale_features(self, feature_map, x_aux):
        """method to recalibrate feature map x"""

    def forward(self, feature_map, x_aux):

        if self.location == 0:
            feature_map = self.rescale_features(feature_map, x_aux)
        residual = feature_map

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 2:
            feature_map = self.rescale_features(feature_map, x_aux)
        out = self.conv1(feature_map)
        out = self.bn1(out)

        if self.location == 3:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        if self.location == 4:
            out = self.rescale_features(out, x_aux)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)

        return out


class FilmBlock(FilmBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
        stride: int = 2,
        ndim_non_img: int = 11,  # Adapt it to the data
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 7,
    ):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        self.bottleneck_dim = bottleneck_dim
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None
        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def rescale_features(self, feature_map, x_aux):

        attention = self.aux(x_aux)

        assert (attention.size(0) == feature_map.size(0)) and (
            attention.dim() == 2
        ), f"Invalid size of output tensor of auxiliary network: {attention.size()}"

        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1,
                                   1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1,
                                   1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1,
                                   1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1,
                                   1).expand_as(feature_map)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift


class Film(BaseModel):
    """
    adapted version of Perez et al. (AAAI, 2018)
    https://arxiv.org/abs/1709.07871
    """

    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        filmblock_args: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(
            in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(
            n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(
            n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(
            2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = FilmBlock(4 * n_basefilters, 8 * n_basefilters,
                                bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class DAFTBlock(FilmBase):
    # Block for ZeCatNet
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
        stride: int = 2,
        ndim_non_img: int = 11,  # was 15 before
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 32,
    ) -> None:

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        self.bottleneck_dim = bottleneck_dim
        aux_input_dims = self.film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img +
             aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def rescale_features(self, feature_map, x_aux):

        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1,
                                   1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1,
                                   1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1,
                                   1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1,
                                   1).expand_as(feature_map)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift


class DAFT(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        filmblock_args: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(
            in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(
            n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(
            n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(
            2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = DAFTBlock(4 * n_basefilters, 8 * n_basefilters,
                                bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
