"""
`SamplerQNN` – A flexible classical neural sampler.

Improvements over the seed:
* Configurable input/output dimensionality and hidden layer sizes.
* Optional dropout for regularisation.
* Explicit `sample` method that draws discrete samples from the learned probability distribution.
* Custom weight initialization for better convergence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A probabilistic neural sampler that maps an *input* vector to a categorical distribution
    over *output_dim* classes.  The network can be stacked with any depth and optional
    dropout, making it suitable for both toy experiments and larger-scale tasks.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    output_dim : int
        Number of output classes.
    hidden_dims : Sequence[int]
        Sizes of intermediate hidden layers.  Defaults to a single hidden layer of size 4.
    dropout : float, optional
        Dropout probability applied after every hidden layer.  Set to 0.0 to disable.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_dims: list[int] | tuple[int,...] = (4,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hid in hidden_dims:
            layers.append(nn.Linear(prev_dim, hid))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hid
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """He‑init for linear layers followed by Tanh activations."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="tanh")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability vector over output classes."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw discrete samples from the categorical distribution produced by the network.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input vectors of shape ``(batch, input_dim)``.
        n_samples : int
            Number of samples to draw per input.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, n_samples)`` containing class indices.
        """
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples=n_samples, replacement=True)


__all__ = ["SamplerQNN"]
