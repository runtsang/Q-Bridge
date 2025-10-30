"""SamplerQNN – classical neural network implementation.

This module extends the original two‑layer design by adding:
* Two hidden layers with ReLU activations.
* Dropout for regularisation.
* A `train` helper that accepts a DataLoader and optimises
  the cross‑entropy loss with Adam.
* A `sample` method that returns the softmax probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class SamplerQNN(nn.Module):
    """
    A two‑hidden‑layer Sampler network with dropout.
    
    Parameters
    ----------
    input_dim : int, default=2
        Dimension of the input feature vector.
    hidden_dims : tuple[int, int], default=(64, 32)
        Sizes of the two hidden layers.
    dropout_prob : float, default=0.2
        Dropout probability applied after each hidden layer.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[1], 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning the probability distribution."""
        return self.forward(x)

    def train(
        self,
        loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        """
        Train the network on a given DataLoader.

        Parameters
        ----------
        loader : DataLoader
            DataLoader yielding (inputs, targets) where targets are
            class indices 0 or 1.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate for Adam optimiser.
        device : str | None
            Device to run training on; defaults to CUDA if available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = self(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
