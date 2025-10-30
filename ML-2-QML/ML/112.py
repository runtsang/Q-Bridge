"""Enhanced sampler network with dropout, batch normalization and a deeper architecture.

The class inherits from the original SamplerQNN and extends it with additional
layers to improve expressivity and regularisation. It also exposes a `sample`
method that returns a probability distribution for a given input.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen122(nn.Module):
    """
    A deeper neural sampler network with dropout and batch normalisation.
    Input dimension is 2, output dimension is 2 (softmax probabilities).
    """

    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int,...] = (8, 4), dropout: float = 0.1) -> None:
        """
        Parameters
        ----------
        input_dim: int
            Size of input vector.
        hidden_dims: tuple[int,...]
            Sizes of hidden layers.
        dropout: float
            Dropout probability applied after each hidden layer.
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from the probability distribution produced by the network.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (..., 2).

        Returns
        -------
        torch.Tensor
            Sampled one‑hot vectors of the same shape as `x`.
        """
        probs = self.forward(x)
        # use multinomial sampling
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

__all__ = ["SamplerQNNGen122"]
