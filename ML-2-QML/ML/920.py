"""Enhanced classical sampler network with training and sampling utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

__all__ = ["SamplerQNN"]


class SamplerQNN(nn.Module):
    """
    A two‑layer MLP that maps a 2‑dimensional input to a categorical distribution over 2 classes.
    The network includes batch‑normalisation, dropout, and exposes convenient training and sampling
    methods for quick experimentation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector. Default is 2.
    hidden_dim : int
        Size of the hidden layer. Default is 8.
    output_dim : int
        Number of output classes. Default is 2.
    dropout : float
        Drop‑out probability applied after the hidden layer. Default is 0.2.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        output_dim: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute the softmax probability vector."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def train_model(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        epochs: int = 20,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Simple training loop that accepts a DataLoader, a loss function and an optimizer.

        Parameters
        ----------
        dataloader : DataLoader
            Iterable over (input, target) pairs.
        loss_fn : nn.Module
            Loss function that accepts predictions and targets.
        optimizer : Optimizer
            Optimiser that updates the model parameters.
        epochs : int
            Number of training epochs.
        device : torch.device | str
            Device to run the training on.
        """
        self.to(device)
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    def sample(self, x: torch.Tensor, mode: str = "categorical") -> torch.Tensor:
        """
        Generate samples from the output distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).
        mode : str
            Sampling strategy: 'categorical' returns the argmax,'multinomial' draws from the softmax.

        Returns
        -------
        torch.Tensor
            Sampled class indices of shape (batch,).
        """
        probs = self(x)
        if mode == "categorical":
            return torch.argmax(probs, dim=-1)
        elif mode == "multinomial":
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            raise ValueError("mode must be 'categorical' or'multinomial'")
