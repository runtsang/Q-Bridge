"""HybridFCL – a multi‑layer classical fully connected module with dropout.

The original seed implemented a single linear layer and a trivial
``run`` method.  This extension allows the user to specify an arbitrary
layer stack, add dropout for regularisation, and expose a PyTorch
``forward`` that can be used in a standard training loop.
"""

__all__ = ["HybridFCL"]

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence


class HybridFCL(nn.Module):
    """Multi‑layer feed‑forward network with optional dropout.

    Parameters
    ----------
    layer_sizes : Sequence[int]
        A sequence of integers defining the sizes of each hidden layer.
        The first element is the input dimensionality, the last element
        is the output dimensionality.
    dropout : float, optional
        Dropout probability applied after every hidden layer.
        Default is 0.0 (no dropout).
    """

    def __init__(self, layer_sizes: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes")

        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            # Use tanh non‑linearity to mirror the original seed
            layers.append(nn.Tanh())

        # Remove the last non‑linearity to keep the interface similar
        layers.pop()

        self.network = nn.Sequential(*layers)

    def forward(self, thetas: Iterable[float]) -> np.ndarray:
        """Compute the network output for a batch of parameter vectors.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of floating‑point parameters; the first ``len(thetas)`` values
            are fed into the network as a column vector.  The method returns
            a NumPy array of shape ``(1,)`` containing the mean of the
            network outputs across the batch, mimicking the original
            ``run`` signature.

        Returns
        -------
        np.ndarray
            Array of shape ``(1,)`` with the mean output value.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        # Forward pass
        out = self.network(values)
        # Return mean over the batch dimension
        expectation = out.mean(dim=0)
        return expectation.detach().numpy()
