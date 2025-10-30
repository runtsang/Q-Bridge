import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Classical quanvolution filter that processes 2x2 patches."""
    def __init__(self) -> None:
        super().__init__()
        # 1 input channel -> 4 output channels
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W)
        features = self.conv(x)
        # flatten patches
        return features.view(x.size(0), -1)

class QuanvolutionRegressor(nn.Module):
    """Classical regression model that uses quanvolution filter and a linear head."""
    def __init__(self, num_features: int = 4 * 14 * 14) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        out = self.regressor(features)
        return out.squeeze(-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionRegressor"]
