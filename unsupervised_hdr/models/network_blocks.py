from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn


@dataclass
class FeatureInfo:
    c: List[int] = field(default_factory=list)
    r: List[int] = field(default_factory=list)

    def channels(self):
        return self.c

    def reduction(self):
        return self.r


class ConcatPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.mp = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
        )
        self.ap = nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
        )

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Conv2dReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_batchnorm: bool = False,
        **batchnorm_params
    ):

        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=not (use_batchnorm),
            ),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
