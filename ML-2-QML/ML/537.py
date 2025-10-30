"""
Classical sampler network with residual connections and regularization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNGen180(nn.Module):
    """
    Residual feed‑forward sampler.

    Architecture:
        - Input layer: 2 → 8
        - Residual block: 8 → 8
        - Dropout (p=0.2)
        - Output layer: 8 → 2
        - Softmax over last dimension
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        # Residual block
        self.res_block = nn.Sequential(
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=0.2)
        self.output_layer = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass with residual addition.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., 2).

        Returns
        -------
        torch.Tensor
            Probability distribution over 2 classes.
        """
        out = self.input_layer(x)
        res = self.res_block(out)
        out = out + res  # Residual addition
        out = self.dropout(out)
        out = self.output_layer(out)
        return F.softmax(out, dim=-1)


__all__ = ["SamplerQNNGen180"]
