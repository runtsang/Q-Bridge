"""Hybrid QCNN architecture with a classical backbone.

This module defines :class:`QCNNHybrid`, which implements the
classical convolution‑like layers from the QCNN seed and a dense
sigmoid head.  The network can be used as a drop‑in replacement for
the pure quantum QCNN, or as a baseline for comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """Classical QCNN backbone with a sigmoid head.

    The architecture is a lightweight stack of linear layers that
    mimics the feature‑map, convolution and pooling stages of the
    quantum QCNN circuit.  The final layer produces a single logit
    that is passed through a sigmoid to obtain a probability for
    the positive class.

    Parameters
    ----------
    in_channels : int, default 3
        Number of channels in the input image.  The default matches
        the 3‑channel images used in the binary classification seed.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        # Feature map: flatten 3x3 patch and map to 8‑dim
        self.feature_map = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 3 * 3, 8),
            nn.Tanh(),
        )
        self.conv1 = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, channels, 3, 3)``.  The
            3×3 spatial size is chosen to match the feature‑map used
            in the quantum seed.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, 2)`` containing
            ``[prob, 1‑prob]`` for each example.
        """
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        logits = self.head(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

    def predict(self, inputs: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convenience wrapper that returns numpy probabilities."""
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            probs = self.forward(inputs).cpu().numpy()
        return probs

__all__ = ["QCNNHybrid"]
