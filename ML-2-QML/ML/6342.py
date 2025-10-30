"""
Classical sampler network with extended architecture and training utilities.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class SamplerQNN(nn.Module):
    """
    A deeper neural sampler that models a 2‑dimensional probability distribution.
    Includes batch‑norm, dropout, and an optional training method.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Return a probability distribution over the 2‑dimensional space.
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def train_sampler(
        self,
        data: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> None:
        """
        Train the sampler on provided data using cross‑entropy loss.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (N, 2) containing target samples.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        batch_size : int
            Size of minibatches.
        """
        self.train()
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(data).float()
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for batch in loader:
                x = batch[0]
                probs = self(x)
                # Convert continuous samples to discrete class indices
                target = torch.argmax(probs, dim=1)
                loss = criterion(probs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


__all__ = ["SamplerQNN"]
