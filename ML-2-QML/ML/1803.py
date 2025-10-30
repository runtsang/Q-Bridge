"""FullyConnectedLayerGen402 – a classical deep, regularised fully‑connected block.

This class extends the original 1‑D linear layer to a stack of linear layers
with optional dropout and batch‑normalisation, mimicking a small neural
network.  It follows the PyTorch module API, making it drop‑in
compatible with typical training pipelines.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class FullyConnectedLayerGen402(nn.Module):
    """
    A small feed‑forward network that accepts a list of parameters
    (thetas) for each layer.  Parameters are expected to be supplied
    as a flat list; the module internally reshapes them into weights
    and biases for each linear layer.

    Parameters
    ----------
    n_features : int
        Size of the input feature vector.
    n_hidden : int, optional
        Number of hidden layers.  Defaults to 0 (single linear layer).
    hidden_dim : int, optional
        Width of each hidden layer.  Ignored if n_hidden==0.
    dropout : float, optional
        Drop‑out probability applied after each hidden layer.
    batch_norm : bool, optional
        Whether to apply a batch‑norm layer after each hidden layer.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 0,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = [nn.Linear(n_features, hidden_dim) if n_hidden else nn.Linear(n_features, 1)]
        if n_hidden:
            for _ in range(n_hidden - 1):
                seq = [nn.Linear(hidden_dim, hidden_dim)]
                if batch_norm:
                    seq.append(nn.BatchNorm1d(hidden_dim))
                seq.append(nn.ReLU())
                if dropout > 0.0:
                    seq.append(nn.Dropout(dropout))
                layers.extend(seq)
            layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        # Flatten all parameters into a single vector for the `run` API
        self._param_shapes = []
        self._param_sizes = []
        for param in self.net.parameters():
            self._param_shapes.append(param.shape)
            self._param_sizes.append(param.numel())

    def _unflatten_params(self, flat_params: Iterable[float]) -> List[torch.Tensor]:
        """Reshape a flattened parameter list into the
        shapes expected by the network."""
        flat = torch.tensor(list(flat_params), dtype=torch.float32)
        tensors: List[torch.Tensor] = []
        idx = 0
        for shape, size in zip(self._param_shapes, self._param_sizes):
            tensors.append(flat[idx : idx + size].reshape(shape))
            idx += size
        return tensors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.net(x)

    def run(self, thetas: Iterable[float], input_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run the network with a flat list of parameters and an optional
        input vector.  If `input_vec` is None, a dummy vector of zeros
        with the required shape is used.

        Returns
        -------
        output: torch.Tensor
            Model output as a 1‑D tensor of shape (1,).
        """
        # Load external parameters
        for param, new_val in zip(self.net.parameters(), self._unflatten_params(thetas)):
            param.data.copy_(new_val)

        if input_vec is None:
            input_vec = torch.zeros(self.net[0].in_features, dtype=torch.float32)

        return self.forward(input_vec)

__all__ = ["FullyConnectedLayerGen402"]
