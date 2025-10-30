"""Enhanced classical sampler network with configurable depth and regularisation.

The original seed used a fixed 2‑layer MLP.  This version generalises the architecture, adds dropout and batch‑norm, and exposes utilities for weight inspection and state‑dict handling.  It remains fully compatible with PyTorch training pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Sequence, Tuple


class SamplerQNN(nn.Module):
    """
    A multi‑layer perceptron that maps an arbitrary‑dimensional input to a probability
    distribution over two classes.  The network is configurable via *hidden_dims*,
    *dropout*, and *batch_norm* flags.

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector.
    hidden_dims : Sequence[int]
        Sizes of hidden layers.  A single integer is accepted for a single hidden layer.
    dropout : float, optional
        Drop‑out probability applied after each hidden layer.  ``0`` disables dropout.
    batch_norm : bool, default False
        Whether to insert a batch‑normalisation layer after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] | int = (4,),
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        layers: list[nn.Module] = []
        in_dim = input_dim

        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a probability distribution over two classes."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def state_dict(self, *args, **kwargs) -> dict:
        """Convenience wrapper around :py:meth:`torch.nn.Module.state_dict`."""
        return super().state_dict(*args, **kwargs)

    def load_from_dict(self, state: dict) -> None:
        """Load weights from a state dictionary."""
        self.load_state_dict(state)

__all__ = ["SamplerQNN"]
