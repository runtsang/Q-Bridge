"""
Hybrid classical estimator combining regression and classification.
The network architecture is fully configurable: input dimension, hidden size,
depth, and output type (regression or binary classification).  The class
exposes a simple forward pass and a convenience method for computing the
loss, enabling integration into any PyTorch training loop.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Tuple


class EstimatorQNN(nn.Module):
    """
    A flexible feed‑forward network that can perform either regression or binary
    classification.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    hidden_dim : int, default 8
        Width of each hidden layer.
    depth : int, default 3
        Number of hidden layers.
    classification : bool, default False
        When True the network ends with a 2‑unit output and a LogSoftmax
        activation.  When False it produces a single continuous output.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 8,
        depth: int = 3,
        classification: bool = False,
    ) -> None:
        super().__init__()
        self.classification = classification
        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            # Use Tanh for regression, ReLU for classification
            layers.append(nn.Tanh() if not classification else nn.ReLU())
            in_dim = hidden_dim
        # Final head
        out_dim = 2 if classification else 1
        layers.append(nn.Linear(in_dim, out_dim))
        if classification:
            layers.append(nn.LogSoftmax(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch_size, 1) or
            classification logits of shape (batch_size, 2).
        """
        return self.net(x)

    def loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        *,  # Force keyword arguments for clarity
        reg: bool = False,
    ) -> Tensor:
        """
        Compute loss between predictions and targets.

        Parameters
        ----------
        predictions : torch.Tensor
            Output from the network.
        targets : torch.Tensor
            Ground‑truth values.  For regression this should be a scalar
            per sample; for classification it should contain class indices.
        reg : bool, default False
            Whether to use MSE (True) or cross‑entropy (False).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if self.classification:
            return F.nll_loss(predictions, targets.long(), reduction="mean")
        else:
            # For regression we use MSE
            return F.mse_loss(predictions.squeeze(-1), targets.float(), reduction="mean")


__all__ = ["EstimatorQNN"]
