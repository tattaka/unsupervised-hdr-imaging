import torch
from torch import nn
from torch.nn import functional as F


class ImageSpaceLoss(nn.Module):
    def __init__(self, lam: int = 5) -> None:
        super().__init__()
        self.lam = lam
        self.l1_loss = nn.L1Loss()
        # TODO: Try revisiting L1 loss(https://arxiv.org/abs/2201.10084)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        loss = self.l1_loss(x1, x2) + self.lam * (
            1 - torch.mean(F.cosine_similarity(x1, x2, dim=1))
        )
        return loss
