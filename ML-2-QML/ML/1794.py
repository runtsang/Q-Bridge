"""Enhanced classical Sampler Network for classification and sampling.

This module extends the original 2‑layer linear network by adding:
* A configurable hidden size and two hidden layers.
* Batch‑normalisation and dropout for regularisation.
* A convenient `sample` method that returns class indices drawn from
  the learned probability distribution.

The interface mirrors the original `SamplerQNN` class so that legacy
code can swap the implementation without changes to the calling
scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A lightweight neural sampler that maps 2‑dimensional inputs to a
    probability vector over 2 classes.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_dim : int, default 8
        Number of units in the intermediate hidden layers.
    dropout : float, default 0.2
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the
        network's output probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch, input_dim).
        num_samples : int
            Number of samples to draw per input.

        Returns
        -------
        torch.Tensor
            Sample indices of shape (batch, num_samples).
        """
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples, replacement=True)


__all__ = ["SamplerQNN"]
