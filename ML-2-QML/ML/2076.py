"""Enhanced fully‑connected layer with trainable weights and optional regularisation.

The class can be instantiated with arbitrary depth and will expose a ``run`` method
that accepts a sequence of parameters and returns a single expectation‑value‑style
tensor, mimicking the behaviour of the original seed but with richer dynamics.
"""

import torch
from torch import nn
import numpy as np
from typing import Iterable, Sequence

class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: Sequence[int] | None = None,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        """Build a stack of linear layers.

        Parameters
        ----------
        n_features: int
            Size of the input vector.
        hidden_sizes: Sequence[int] | None
            Sizes of hidden layers. ``None`` creates a single output layer.
        dropout: float
            Dropout probability applied after each hidden layer.
        use_batchnorm: bool
            Whether to insert a BatchNorm1d after each linear layer.
        """
        super().__init__()
        layers = []
        in_size = n_features
        hidden_sizes = hidden_sizes or [1]
        for out_size in hidden_sizes:
            layers.append(nn.Linear(in_size, out_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_size))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_size = out_size
        # Final linear mapping to a single scalar
        layers.append(nn.Linear(in_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.model(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the network on a sequence of scalars.

        The input is interpreted as a batch of one‑dimensional feature vectors,
        each containing a single value. The network processes the batch and
        returns the mean of the final activations as a numpy array, emulating
        the expectation‑value semantics of the original FCL example.
        """
        with torch.no_grad():
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            output = self.forward(values)
            expectation = output.mean(dim=0)
            return expectation.detach().cpu().numpy()
