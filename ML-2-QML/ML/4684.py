"""Hybrid classical QCNN model with regression and classification heads.

The architecture mirrors the quantum‑inspired QCNN but is implemented
entirely in PyTorch.  The network consists of a feature‑map layer,
three convolution‑pool blocks, and two optional output heads:
a regression head (Estimator‑style) and a classification head
(Sampler‑style).  The model can be used in either mode by
specifying the `task` argument in the forward pass.

The design demonstrates how classical layers can emulate quantum
operations and how different output objectives can be combined in a
single module.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QCNNHybrid(nn.Module):
    """Convolution‑pool network with regression and classification heads."""

    def __init__(self) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh()
        )
        # Convolution‑pool blocks
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Output heads
        self.regressor = nn.Sequential(
            nn.Linear(4, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(4, 4), nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, inputs: torch.Tensor, task: str = "regression") -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, 8).
        task : str, optional
            One of ``'regression'`` or ``'classification'``.
            Defaults to ``'regression'``.
        """
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        if task == "regression":
            return torch.sigmoid(self.regressor(x))
        elif task == "classification":
            return F.softmax(self.classifier(x), dim=-1)
        else:
            raise ValueError(f"Unknown task: {task}")

__all__ = ["QCNNHybrid"]
