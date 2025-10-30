"""Enhanced classical quanvolution filter with contrastive pre‑training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

@dataclass
class ContrastiveLoss:
    """NT-Xent loss used for self‑supervised pre‑training."""
    temperature: float = 0.5
    eps: float = 1e-12

    def __call__(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.mm(z1, z2.t())
        logits = logits / self.temperature
        N = logits.size(0)
        mask = torch.eye(N, dtype=torch.bool, device=logits.device)
        logits_mask = torch.ones_like(mask, dtype=torch.bool) & ~mask
        exp_logits = torch.exp(logits) * logits_mask
        denom = exp_logits.sum(dim=1, keepdim=True)
        pos_logits = torch.diag(logits)
        loss = -torch.log((torch.exp(pos_logits) + self.eps) / (denom.squeeze() + self.eps))
        return loss.mean()


class QuanvolutionFilter(nn.Module):
    """Classical quanvolution filter that replaces a single 2‑D convolution with a learnable block
    that generates multiple feature maps. The output channel dimension is increased to 8,
    providing richer representations for downstream processing.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 8, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_conv(x)
        features = self.proj(features)
        return features.view(features.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Full model for supervised classification that uses the quanvolution filter
    followed by a small classical CNN head and a final linear layer.
    A contrastive loss can be computed on the output of the filter for self‑supervised pre‑training.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.head = nn.Sequential(
            nn.Linear(8 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        hidden = self.head(features)
        logits = self.classifier(hidden)
        return F.log_softmax(logits, dim=-1)

    def pretrain_step(self, x: torch.Tensor, loss_fn: Optional[ContrastiveLoss] = None) -> Tensor:
        if loss_fn is None:
            loss_fn = ContrastiveLoss()
        aug1 = torch.flip(x, dims=[-1])
        aug2 = torch.roll(x, shifts=2, dims=[-1])
        z1 = self.qfilter(aug1)
        z2 = self.qfilter(aug2)
        return loss_fn(z1, z2)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier", "ContrastiveLoss"]
