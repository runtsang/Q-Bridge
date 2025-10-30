"""
A robust classical sampler network built on PyTorch.

Features:
* Two hidden layers with configurable widths.
* Batch‑normalization and dropout for regularisation.
* `sample` method that draws categorical samples from the softmax output.
* Flexible input dimension (default 2) to support higher‑dimensional data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    Classical sampler neural network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input vector.
    hidden_dims : list[int], default [4, 4]
        Width of each hidden layer.
    dropout_rate : float, default 0.1
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [4, 4]
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a probability distribution over two classes.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape ``(batch, 2)``.
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def log_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return the log‑probability of the forward pass.

        This is useful for likelihood‑based training objectives.
        """
        probs = self.forward(inputs)
        return torch.log(probs + 1e-12)

    def sample(self, inputs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw categorical samples from the output distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape ``(batch, input_dim)``.
        n_samples : int, default 1
            Number of samples per input.

        Returns
        -------
        torch.Tensor
            Integer samples of shape ``(batch, n_samples)``.
        """
        probs = self.forward(inputs)
        # Create a categorical distribution per batch element
        dist = torch.distributions.Categorical(probs)
        return dist.sample((n_samples,)).transpose(0, 1)

__all__ = ["SamplerQNN"]
