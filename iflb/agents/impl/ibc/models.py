import enum
from typing import Sequence
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .utils import normalize
from .modules import CoordConv, GlobalAvgPool2d, SpatialSoftArgmax, GlobalMaxPool2d

def get_layer(input_dim, output_dim, spectral_norm=False):
    if spectral_norm:
        return nn.utils.spectral_norm(nn.Linear(input_dim, output_dim).float())
    else:
        return nn.Linear(input_dim, output_dim).float()

class ActivationType(enum.Enum):
    RELU = nn.ReLU
    SELU = nn.SiLU

class SpatialReduction(enum.Enum):
    SPATIAL_SOFTMAX = SpatialSoftArgmax
    AVERAGE_POOL = GlobalAvgPool2d
    MAX_POOL = GlobalMaxPool2d

@dataclasses.dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int = 256
    output_dim: int = 1
    hidden_layers: int = 6
    activation_fn: ActivationType = ActivationType.RELU
    spectral_norm: bool = False
    net_type: str = 'mlp'
    normalize_inputs: bool = False

@dataclasses.dataclass(frozen=True)
class CNNConfig:
    in_channels: int = 3
    blocks: Sequence[int] = dataclasses.field(default=(32, 64, 128, 256))
    activation_fn: ActivationType = ActivationType.RELU

@dataclasses.dataclass(frozen=True)
class ConvMLPConfig:
    cnn_config: CNNConfig
    mlp_config: MLPConfig
    spatial_reduction: SpatialReduction = SpatialReduction.SPATIAL_SOFTMAX
    coord_conv: bool = False

class MLP(nn.Module):
    """A feedforward multi-layer perceptron."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        layers: Sequence[nn.Module]
        if config.hidden_layers == 0:
            layers = [nn.Linear(config.input_dim, config.output_dim)]
        else:
            layers = [
                get_layer(config.input_dim, config.hidden_dim, config.spectral_norm),
                config.activation_fn.value(),
            ]
            for _ in range(config.hidden_layers - 1):
                layers += [
                    get_layer(config.hidden_dim, config.hidden_dim, config.spectral_norm),
                    config.activation_fn.value(),
                ]
            layers += [get_layer(config.hidden_dim, config.output_dim)]

        self.net = nn.Sequential(*layers).float()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        assert config.hidden_layers % 2 == 0

        self.fc0 = get_layer(config.input_dim, config.hidden_dim, config.spectral_norm)
        self.activation0 = config.activation_fn.value()

        self.layers_1 = nn.ModuleList()
        self.activations_1 = nn.ModuleList()
        self.layers_2 = nn.ModuleList()
        self.activations_2 = nn.ModuleList()

        for i in range(config.hidden_layers // 2):
            self.layers_1.append(get_layer(config.hidden_dim, config.hidden_dim, config.spectral_norm))
            self.activations_1.append(config.activation_fn.value())

            self.layers_2.append(get_layer(config.hidden_dim, config.hidden_dim, config.spectral_norm))
            self.activations_2.append(config.activation_fn.value())

        self.fc_last = get_layer(config.hidden_dim, config.output_dim)
        
    def forward(self, x):
        raise NotImplementedError()


class ResNetOrig(ResNet):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        x = self.activation0(self.fc0(x))
        for i in range(len(self.layers_1)):
            x_residual = nn.Identity()(x)
            x = self.activations_1[i](self.layers_1[i](x))
            x = self.activations_2[i](self.layers_2[i](x))
            x = x + x_residual
        return self.fc_last(x)


class ResNetPreActivation(ResNet):
    def __init__(self, config):
        super().__init__(config)
        del self.activation0

    def forward(self, x):
        x = self.fc0(x)
        for i in range(len(self.layers_1)):
            x_residual = nn.Identity()(x)
            x = self.layers_1[i](self.activations_1[i](x))
            x = self.layers_2[i](self.activations_2[i](x))
            x = x + x_residual
        return self.fc_last(x)


class EBM(nn.Module):
    
    def __init__(self, config: DictConfig):
        super().__init__()
        net_type = config.net_type.lower()
        if net_type == 'mlp':
            self.net = MLP(config)
        elif net_type == 'resnet':
            self.net = ResNetOrig(config)
        elif net_type == 'resnet_pre':
            self.net = ResNetPreActivation(config)
        self.data_statistics = None
        self.normalize = config.normalize_inputs

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.normalize and self.data_statistics:
            x = normalize(x, self.data_statistics['x_mean'], self.data_statistics['x_std'])
            y = normalize(y, self.data_statistics['y_mean'], self.data_statistics['y_std'])   
        if x.size() == y.size():
            fused = torch.cat([x, y], dim=-1).unsqueeze(1)
        else:
            fused = torch.cat([x.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        B, N, D = fused.size()
        out = self.net(fused)
        return out.view(B, N)

    def update_stats(self, x_mu, x_sigma, y_mu, y_sigma):
        self.data_statistics = {
            'x_mean': x_mu,
            'x_std': x_sigma, 
            'y_mean': y_mu,
            'y_std': y_sigma
        }

class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        activation_fn: ActivationType,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.activation = activation_fn.value()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(x)
        out = self.conv2(out)
        return out + x


class CNN(nn.Module):
    """A residual convolutional network."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()

        depth_in = config.in_channels

        layers = []
        for depth_out in config.blocks:
            layers.extend(
                [
                    nn.Conv2d(depth_in, depth_out, 3, padding=1),
                    ResidualConvBlock(depth_out, config.activation_fn),
                ]
            )
            depth_in = depth_out

        self.net = nn.Sequential(*layers)
        self.activation = config.activation_fn.value()

    def forward(self, x: torch.Tensor, activate: bool = False) -> torch.Tensor:
        out = self.net(x)
        if activate:
            return self.activation(out)
        return out


class ConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        out = F.relu(self.conv(out))
        out = self.reducer(out)
        out = self.mlp(out)
        return out


class EBMConvMLP(nn.Module):
    def __init__(self, config: ConvMLPConfig) -> None:
        super().__init__()

        self.coord_conv = config.coord_conv

        self.cnn = CNN(config.cnn_config)
        self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
        self.reducer = config.spatial_reduction.value()
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.coord_conv:
            x = CoordConv()(x)
        out = self.cnn(x, activate=True)
        out = F.relu(self.conv(out))
        out = self.reducer(out)
        fused = torch.cat([out.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.mlp(fused)
        return out.view(B, N)
