from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolutional filter producing 4 feature maps."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Classical classifier using a quanvolution filter followed by a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

class ClassicalFullyConnectedLayer(nn.Module):
    """Simulated quantum fully‑connected layer implemented with a linear net and tanh."""
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, n_features)
        expectation = torch.tanh(self.linear(x)).mean(dim=1, keepdim=True)
        return expectation

class QuanvolutionHybrid(nn.Module):
    """Hybrid model: classical quanvolution filter + simulated quantum fully‑connected layer."""
    def __init__(self, n_fc_features: int = 128) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qfc = ClassicalFullyConnectedLayer(n_fc_features)
        self.out = nn.Linear(1, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # (batch, 4*14*14)
        params = features[:, :self.qfc.linear.in_features]
        q_out = self.qfc(params)    # (batch, 1)
        logits = self.out(q_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier", "QuanvolutionHybrid"]
