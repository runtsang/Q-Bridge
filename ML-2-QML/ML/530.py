"""
Hybrid classical‑quantum Quanvolution network with extended capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class QuanvolutionEnhanced(nn.Module):
    """
    Classical Quanvolution network with optional attention weighting.
    """

    def __init__(self, use_attention: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(4 * 14 * 14, 4 * 14 * 14),
                nn.Softmax(dim=-1)
            )
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: Tensor) -> Tensor:
        features = self.conv(x).view(x.size(0), -1)
        if self.use_attention:
            weights = self.attention(features)
            features = features * weights
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionEnhanced"]
