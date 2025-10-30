"""Hybrid regression model combining classical feed‑forward, optional LSTM, and a simple sampler network.

The module mirrors the original QuantumRegression seed but adds:
* an optional `nn.LSTM` for sequential data (inspired by QLSTM),
* a lightweight `SamplerQNN` that turns the two‑dimensional output into a probability distribution,
* and a consistent API that can be swapped with the quantum counterpart.

All components are fully PyTorch‑compatible and can be trained with standard optimisers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def SamplerQNN() -> nn.Module:
    """Classical sampler that maps a 2‑dimensional vector to a probability distribution."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


class HybridRegression(nn.Module):
    """
    A regression head that can optionally process sequential data with an LSTM.
    The final output is a 2‑dimensional probability distribution returned by a
    small sampler network.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 32,
        use_lstm: bool = False,
        lstm_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.use_lstm = use_lstm

        if self.use_lstm:
            self.lstm = nn.LSTM(num_features, lstm_hidden_dim, batch_first=True)
            self.head = nn.Linear(lstm_hidden_dim, 2)
        else:
            self.net = nn.Sequential(
                nn.Linear(num_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
            )

        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            For a single step regression: shape (batch, num_features).
            For sequential regression: shape (batch, seq_len, num_features).

        Returns
        -------
        torch.Tensor
            A probability distribution over two outputs.
        """
        if self.use_lstm:
            out, _ = self.lstm(x)
            out = out[:, -1, :]  # last time step
            logits = self.head(out)
        else:
            logits = self.net(x)

        probs = self.sampler(logits)
        return probs


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data", "SamplerQNN"]
