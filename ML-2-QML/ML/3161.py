"""Hybrid classical network that combines convolutional feature extraction with a shallow feed‑forward head.

The architecture is inspired by the original Quanvolution example and the EstimatorQNN regressor.
It first applies a 2‑D convolution that reduces each 2×2 image patch to a 4‑dimensional feature map,
then flattens the feature map and feeds it into a shallow neural network consisting of
Linear → Tanh → Linear → Tanh → Linear layers. This design keeps the model lightweight
while allowing the head to learn non‑linear mappings from the extracted features.

The module is fully compatible with PyTorch and can be used as a drop‑in replacement
for the original ``QuanvolutionClassifier`` in the ``Quanvolution.py`` file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuanvolutionNet(nn.Module):
    """
    Classical hybrid network that mimics the behaviour of a quanvolution filter
    followed by a simple feed‑forward classifier.
    """
    def __init__(self, in_channels: int = 1, out_classes: int = 10) -> None:
        super().__init__()
        # 2×2 patches → 4‑dimensional feature map
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        # Flatten and feed‑forward head
        self.head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, out_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Log‑softmax probabilities of shape (batch, out_classes).
        """
        features = self.conv(x)
        # Flatten the feature map
        features = features.view(x.size(0), -1)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)

# Backward‑compatibility aliases
QuanvolutionFilter = HybridQuanvolutionNet
QuanvolutionClassifier = HybridQuanvolutionNet

__all__ = ["HybridQuanvolutionNet", "QuanvolutionFilter", "QuanvolutionClassifier"]
