"""Combined classical estimator that integrates FCL feature extraction and a linear head.

This module merges the ideas from EstimatorQNN and FCL: a lightweight fully‑connected
feature extractor followed by a small neural head.  It can be trained with standard
PyTorch optimizers and gradients flow through both components.

Usage
-----
>>> from EstimatorQNN__gen024 import EstimatorQNN
>>> model = EstimatorQNN()
>>> x = torch.randn(5, 2)
>>> y = model(x)
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable

class FCL(nn.Module):
    """
    Tiny fully‑connected layer that emulates the quantum FCL example.
    The layer applies a linear transform followed by a tanh activation and
    returns the mean over the feature dimension.  It accepts batched input
    and returns a tensor of shape (batch, 1).
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas: (batch, n_features)
        return torch.tanh(self.linear(thetas)).mean(dim=1, keepdim=True)


class EstimatorQNNCombined(nn.Module):
    """
    Classical estimator that combines an FCL feature extractor with a
    two‑layer neural head.  The design mirrors the EstimatorQNN example
    while using the lightweight FCL sub‑module for feature extraction.
    """
    def __init__(self, n_features: int = 2, hidden_dim: int = 8) -> None:
        super().__init__()
        self.fcl = FCL(n_features)
        self.head = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs
            Tensor of shape (batch, n_features).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        features = self.fcl(inputs)          # shape: (batch, 1)
        output = self.head(features)         # shape: (batch, 1)
        return output


def EstimatorQNN() -> EstimatorQNNCombined:
    """
    Compatibility wrapper that returns the combined estimator.
    """
    return EstimatorQNNCombined()


__all__ = ["EstimatorQNN", "EstimatorQNNCombined"]
