from typing import List

import torch
from torch import nn

from .network_blocks import ConcatPool2d, Conv2dReLU, FeatureInfo


class SimpleEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.ModuleList()
        self.feature_info = FeatureInfo()
        self.conv.append(
            Conv2dReLU(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                use_batchnorm=False,
            )
        )
        self.feature_info.c.append(64)
        self.feature_info.r.append(1)
        self.conv.append(
            nn.Sequential(
                ConcatPool2d(kernel_size=2),
                Conv2dReLU(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_batchnorm=False,
                ),
            )
        )
        self.feature_info.c.append(128)
        self.feature_info.r.append(2)
        self.conv.append(
            nn.Sequential(
                ConcatPool2d(kernel_size=2),
                Conv2dReLU(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_batchnorm=False,
                ),
            )
        )
        self.feature_info.c.append(256)
        self.feature_info.r.append(4)
        self.conv.append(
            nn.Sequential(
                ConcatPool2d(kernel_size=2),
                Conv2dReLU(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_batchnorm=False,
                ),
            )
        )
        self.feature_info.c.append(512)
        self.feature_info.r.append(8)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        y = [x]
        for m in self.conv:
            y.append(m(y[-1]))
        del y[0]
        return y


# if __name__ == "__main__":
#     import torch

#     img = torch.randn(2, 3, 128, 128)
#     e = SimpleEncoder()
#     for y in e(img):
#         print(y.shape)
