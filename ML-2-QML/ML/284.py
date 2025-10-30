"""Enhanced classical sampler network with residual and dropout layers.

This module defines SamplerModule, a lightweight neural network that maps a 2‑dimensional
input to a 2‑dimensional probability vector.  The architecture extends the original
version by adding a residual connection, batch‑normalization and dropout, which
improve generalisation on small datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerModule(nn.Module):
    """
    A 2→2 sampler network with a residual connection.

    Parameters
    ----------
    dropout_rate : float, optional
        Dropout probability applied after the first hidden layer.
    """
    def __init__(self, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(dropout_rate),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 2)
        )
        self.residual = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a probability distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (*, 2).

        Returns
        -------
        torch.Tensor
            Softmaxed logits of shape (*, 2).
        """
        out = self.net(inputs)
        out = out + self.residual(inputs)  # residual addition
        return F.softmax(out, dim=-1)


def SamplerQNN() -> SamplerModule:
    """
    Factory returning the upgraded classical sampler module.
    """
    return SamplerModule()


__all__ = ["SamplerQNN"]
