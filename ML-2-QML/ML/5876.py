"""Enhanced classical estimator with residual blocks and regularisation."""
from __future__ import annotations

import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """
    A regression network featuring:
    * Multiple fully‑connected layers with BatchNorm and ReLU.
    * Dropout for regularisation.
    * A simple residual connection between the input and the first hidden layer.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32, 16]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.1))
            prev_dim = h
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual shortcut
        shortcut = x
        x = self.feature_extractor(x)
        # Add shortcut to the first hidden layer output
        if shortcut.shape == x.shape:
            x = x + shortcut
        return self.output_layer(x)

def EstimatorQNN() -> EstimatorQNN:
    """
    Factory function that returns an instance of the enhanced estimator.
    """
    return EstimatorQNN()

__all__ = ["EstimatorQNN"]
