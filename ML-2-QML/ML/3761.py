"""Hybrid classical sampler network combining CNN feature extraction and 4‑class softmax.

The architecture is a direct fusion of the SamplerQNN linear model and the
QFCModel convolutional backbone.  It extracts image features with a small
CNN, projects them to 4 logits, normalises with BatchNorm1d and produces a
probability distribution via softmax.  The design is deliberately kept
light‑weight to allow seamless swapping with a quantum counterpart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQNN(nn.Module):
    """Classical hybrid sampler network.

    Parameters
    ----------
    input_channels : int, default 1
        Number of channels in the input image.
    image_size : tuple[int, int], default (28, 28)
        Spatial dimensions of the input image.
    """

    def __init__(self, input_channels: int = 1, image_size: tuple[int, int] = (28, 28)) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Calculate the flattened feature dimension
        dummy = torch.zeros(1, input_channels, *image_size)
        out_feat = self.feature_extractor(dummy)
        flat_dim = out_feat.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.feature_extractor(x)
        flattened = features.view(bsz, -1)
        logits = self.fc(flattened)
        out = self.norm(logits)
        return F.softmax(out, dim=-1)


__all__ = ["HybridSamplerQNN"]
