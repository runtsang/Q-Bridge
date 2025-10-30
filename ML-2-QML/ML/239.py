"""Enhanced classical sampler network with probabilistic output and sampling utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SamplerQNN(nn.Module):
    """
    A two‑layer neural network that outputs a probability distribution over two classes.
    Features:
        * Optional batch normalization and dropout for better regularisation.
        * ``sample`` method to draw samples from the output distribution.
        * ``log_likelihood`` for training with a probabilistic loss.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        dropout: float | None = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over the output classes."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1, seed: int | None = None) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the network output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, input_dim)``.
        n_samples : int
            Number of samples to draw per input.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Sample indices of shape ``(batch, n_samples)``.
        """
        probs = self.forward(x)
        if seed is not None:
            torch.manual_seed(seed)
        return torch.multinomial(probs, num_samples=n_samples, replacement=True)

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the log‑likelihood of the true labels ``y`` under the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            True labels (int indices).

        Returns
        -------
        torch.Tensor
            Log‑likelihood loss (negative log‑likelihood).
        """
        probs = self.forward(x)
        log_probs = torch.log(probs + 1e-12)
        return -log_probs.gather(-1, y.unsqueeze(-1)).mean()
