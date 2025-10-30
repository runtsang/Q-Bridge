"""
Enhanced classical sampler network with training and sampling utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SamplerQNN(nn.Module):
    """
    A two‑input, two‑output softmax sampler with optional regularisation and
    built‑in training utilities.

    Parameters
    ----------
    hidden_dim : int, optional
        Size of the hidden layer. Default is 8.
    dropout_rate : float, optional
        Dropout probability applied to the hidden layer. Default is 0.0.
    """

    def __init__(self, hidden_dim: int = 8, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a probability distribution over two classes.
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def train_on_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
    ) -> float:
        """
        Single‑batch training step returning the loss value.
        """
        self.train()
        optimizer.zero_grad()
        probs = self.forward(inputs)
        loss = loss_fn(probs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def sample(
        self,
        num_samples: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """
        Draw samples from the learned distribution.

        Returns
        -------
        samples : torch.Tensor
            Tensor of shape (num_samples, 2) containing one‑hot encoded samples.
        """
        self.eval()
        with torch.no_grad():
            # Create a batch of zeros to feed the network
            dummy_inputs = torch.zeros(num_samples, 2, device=device)
            probs = self.forward(dummy_inputs)
            # Sample from categorical distribution
            samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
            one_hot = F.one_hot(samples, num_classes=2).float()
        return one_hot

    def fit(
        self,
        dataset: TensorDataset,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Convenience training loop over an entire dataset.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                self.train_on_batch(xb, yb, optimizer)


__all__ = ["SamplerQNN"]
