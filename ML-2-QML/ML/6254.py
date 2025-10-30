"""samplerqnn__gen059.py – Classical sampler network with extended capabilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["SamplerQNN"]


class SamplerQNN(nn.Module):
    """
    A deep neural sampler that maps 2‑dimensional inputs to a 2‑class probability distribution.
    The architecture contains two hidden layers, batch‑normalisation, dropout, and a
    flexible sampling method for inference or downstream hybrid workflows.
    """

    def __init__(self, hidden_dims: tuple[int, int] = (64, 32), dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability vector for each input sample."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor | list[list[float]]) -> torch.Tensor:
        """
        Sample from the categorical distribution produced by the network.

        Parameters
        ----------
        inputs: torch.Tensor or list
            Shape (N, 2) or (N, 2) Python list. Converted to a tensor on the same device.

        Returns
        -------
        torch.Tensor
            Integer samples of shape (N,) drawn from the categorical distribution.
        """
        if isinstance(inputs, list):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.net[0].weight.device)
        probs = self.forward(inputs)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def train_on_dataset(
        dataset: TensorDataset,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> "SamplerQNN":
        """
        Convenience training routine that returns a trained model.
        """
        model = SamplerQNN()
        model.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for x, y in dataloader:
                optimizer.zero_grad()
                probs = model(x)
                loss = criterion(probs, y)
                loss.backward()
                optimizer.step()
        return model
