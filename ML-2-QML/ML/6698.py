"""Fully connected layer with classical CNN feature extractor and batch‑norm scaling.

This module mirrors the original FCL but extends it with a two‑layer CNN
feature extractor and dropout for regularisation. The forward pass
produces a batch‑normalised output suitable for downstream tasks.
"""

from __future__ import annotations

import torch
from torch import nn


class FCL(nn.Module):
    """Hybrid classical fully connected layer.

    Parameters
    ----------
    n_features : int, default=1
        Number of input features per sample. Not used directly but kept
        for API compatibility with the original seed.
    hidden_dim : int, default=64
        Size of the hidden linear layer.
    output_dim : int, default=4
        Number of output neurons.
    dropout : float, default=0.1
        Drop‑out probability applied after the hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Two‑stage CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        # Batch‑norm on the output
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalised output of shape (batch, output_dim).
        """
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

    def run(self, thetas: list[float]) -> torch.Tensor:  # pragma: no cover
        """Compatibility shim for the original seed interface."""
        return self.forward(torch.tensor(thetas).view(-1, 1, 28, 28))


def FCL_factory() -> FCL:
    """Convenience factory matching the original ``FCL`` function.

    Returns
    -------
    FCL
        An instance of the hybrid fully connected layer.
    """
    return FCL()


__all__ = ["FCL", "FCL_factory"]
