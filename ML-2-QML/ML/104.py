"""ML implementation of a parameterized fully connected layer with dropout and batch support."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import numpy as np


class ParametricFCL(nn.Module):
    """
    A lightweight fully connected layer that accepts a sequence of parameters
    and returns the mean of a tanh(Linear(param)) transform.
    The module is designed to be drop‑in compatible with the original
    ``FCL`` example but offers additional features:

    * configurable hidden size and dropout for regularisation
    * batch processing of multiple parameter vectors
    * optional device placement (CPU/GPU)
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_units: int = 16,
        dropout: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, hidden_units, bias=True).to(device)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.device = torch.as_tensor([], dtype=torch.float32, device=device).device

    def forward(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of floats representing input parameters.

        Returns
        -------
        np.ndarray
            1‑D array containing the mean activation value.
        """
        # Convert to a tensor and move to the correct device.
        x = torch.as_tensor(list(thetas), dtype=torch.float32, device=self.device).view(-1, 1)
        out = self.linear(x)
        out = self.tanh(out)
        out = self.dropout(out)
        mean_val = out.mean(dim=0)
        return mean_val.detach().cpu().numpy()

    # Alias ``run`` for API compatibility with the original seed.
    run = forward


__all__ = ["ParametricFCL"]
