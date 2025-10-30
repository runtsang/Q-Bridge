"""Hybrid classical convolution + classifier.

This module implements a PyTorch module that emulates the original
``Conv`` interface while adding a configurable feed‑forward head.  The
class can be dropped into any CNN pipeline or used as a stand‑alone
classifier for 2‑D data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ConvClassifier(nn.Module):
    """
    Classical convolutional filter followed by a depth‑controlled
    feed‑forward classifier.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    threshold : float, default 0.0
        Threshold applied after the convolution before the sigmoid.
    depth : int, default 2
        Number of hidden layers in the classifier.
    num_features : int | None, default None
        Dimensionality of the flattened convolution output.  If None,
        it defaults to ``kernel_size ** 2``.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        depth: int = 2,
        num_features: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        if num_features is None:
            num_features = kernel_size ** 2

        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(H, W)`` or ``(B, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, 2)``.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # batch, channel, H, W
        conv_out = self.conv(x)
        act = torch.sigmoid(conv_out - self.threshold)
        flat = act.view(act.size(0), -1)
        return self.classifier(flat)

    def run(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Public API mirroring the legacy ``Conv.run`` method.

        Parameters
        ----------
        data : np.ndarray | torch.Tensor
            2‑D array of shape ``(H, W)`` or a batch.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B, 2)``.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        return self.forward(data)


__all__ = ["ConvClassifier"]
