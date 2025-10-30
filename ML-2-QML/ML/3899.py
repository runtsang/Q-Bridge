"""Classical quanvolution filter and classifier.

This module defines a 2‑D image extractor that mimics a quantum kernel
using a simple 2×2 convolution.  The extracted feature map can be fed
directly into a quantum graph neural network or used on its own for
classification.  The implementation is fully PyTorch‑based.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """
    Classical surrogate of a quantum kernel applied to 2×2 image patches.

    The filter uses a 2×2 convolution with stride 2 to produce a 4‑channel
    feature map.  Each channel corresponds to one of the four pixels in a
    patch and can be interpreted as a quantum input to the QML side.
    """
    def __init__(self) -> None:
        super().__init__()
        # 1 input channel, 4 output channels (one per pixel in a 2×2 patch)
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Flattened feature vector of shape (B, 4 * 14 * 14).
        """
        features = self.conv(x)          # (B, 4, 14, 14)
        return features.view(x.size(0), -1)

    def patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the per‑patch 4‑dim vector for each image.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, 14*14, 4) where each row is the 4‑dim
            representation of a 2×2 patch.
        """
        features = self.conv(x)          # (B, 4, 14, 14)
        return features.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)

class QuanvolutionClassifier(nn.Module):
    """
    End‑to‑end classifier that uses the quanvolution filter followed by
    a linear head.  This can be trained classically or used as a feature
    extractor for quantum downstream models.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class logits.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, 10).
        """
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
