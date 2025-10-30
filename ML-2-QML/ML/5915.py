"""Enhanced fully connected layer with optional trainable linear weights and support for hybrid training.

The class implements a classical fully connected layer that can optionally be used as a stand‑in for a quantum variational circuit.  It accepts an iterable ``thetas`` representing the parameters of the linear layer and returns a scalar expectation value.  The layer can be trained with any PyTorch optimizer and can be combined with a quantum block in a hybrid network.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

class FCL(nn.Module):
    """
    Classical fully‑connected layer.
    Parameters
    ----------
    n_features : int
        Number of input features (default 1).
    use_tanh : bool
        If True apply tanh activation after the linear map.
    """
    def __init__(self, n_features: int = 1, use_tanh: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.use_tanh = use_tanh

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass that mimics the quantum run method.
        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of input parameters.
        Returns
        -------
        torch.Tensor
            Output expectation value as a 1‑D tensor.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.linear(values)
        if self.use_tanh:
            out = torch.tanh(out)
        return out.mean(dim=0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper that returns a NumPy array, matching the original API.
        """
        return self.forward(thetas).detach().numpy()

__all__ = ["FCL"]
