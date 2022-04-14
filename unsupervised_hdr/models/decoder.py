import torch
from torch import nn
from torch.nn import functional as F

from .network_blocks import Conv2dReLU, FeatureInfo


class SimpleDecoder(nn.Module):
    def __init__(self, feature_info: FeatureInfo) -> None:
        super().__init__()
        self.encoder_channel = feature_info.channels()
        self.encoder_reduction = feature_info.reduction()
        self.conv = nn.ModuleList()
        self.conv.append(
            Conv2dReLU(
                in_channels=self.encoder_channel[-1],
                out_channels=self.encoder_channel[-2],
                kernel_size=3,
                stride=1,
                padding=1,
                use_batchnorm=False,
            ),
        )
        self.conv.append(
            Conv2dReLU(
                in_channels=self.encoder_channel[-2] * 2,
                out_channels=self.encoder_channel[-3],
                kernel_size=3,
                stride=1,
                padding=1,
                use_batchnorm=False,
            ),
        )
        self.conv.append(
            Conv2dReLU(
                in_channels=self.encoder_channel[-3] * 2,
                out_channels=self.encoder_channel[-4],
                kernel_size=3,
                stride=1,
                padding=1,
                use_batchnorm=False,
            ),
        )
        self.last_conv = nn.Sequential(
            Conv2dReLU(
                in_channels=self.encoder_channel[-4] * 2,
                out_channels=self.encoder_channel[-4] * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                use_batchnorm=False,
            ),
            nn.Conv2d(
                in_channels=self.encoder_channel[-4] * 2, out_channels=3, kernel_size=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x[-1]
        for i, m in enumerate(self.conv):
            feat = F.interpolate(m(feat), scale_factor=2, mode="bilinear")
            feat = torch.cat([feat, x[-i - 2]], dim=1)
        if self.encoder_reduction[0] > 1:
            feat = F.interpolate(
                feat, scale_factor=self.encoder_reduction[0], mode="bilinear"
            )
        return self.last_conv(feat)


# if __name__ == "__main__":
#     from encoder import SimpleEncoder

#     img = torch.randn(2, 3, 128, 128)
#     e = SimpleEncoder()
#     e_out = e(img)
#     d = SimpleDecoder(e.feature_info)
#     d_out = d(e_out)
#     print(d_out.shape)
