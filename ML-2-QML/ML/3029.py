from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilterClassic(nn.Module):
    """Classical 2×2 convolution followed by flattening."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionFCLClassic(nn.Module):
    """Purely classical classifier using the classic quanvolution filter."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterClassic()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilterClassic", "QuanvolutionFCLClassic"]
