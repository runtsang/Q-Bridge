from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler that merges ideas from QCNN, Quanvolution,
    and EstimatorQNN.  It can operate on either a 2‑dimensional feature
    vector or a single‑channel 28×28 image.
    """
    def __init__(self, use_image: bool = False):
        super().__init__()
        self.use_image = use_image
        if use_image:
            # 2×2 convolution with stride 2 reduces 28×28 to 14×14
            self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
            self.linear = nn.Linear(4 * 14 * 14, 8)
        else:
            self.feature_map = nn.Sequential(nn.Linear(2, 4), nn.Tanh())
            self.linear = nn.Linear(4, 8)

        # QCNN‑style layers
        self.conv1 = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(8, 6), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(6, 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(4, 3), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(3, 2), nn.Tanh())
        # Estimator head
        self.head = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_image:
            x = self.conv(x)          # (B, 4, 14, 14)
            x = x.view(x.size(0), -1) # flatten
            x = self.linear(x)
        else:
            x = self.feature_map(x)
            x = self.linear(x)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        logits = self.head(x)
        return F.softmax(logits, dim=-1)


def SamplerQNN(use_image: bool = False) -> HybridSamplerQNN:
    """Factory returning the configured :class:`HybridSamplerQNN`."""
    return HybridSamplerQNN(use_image=use_image)


__all__ = ["HybridSamplerQNN", "SamplerQNN"]
