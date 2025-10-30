"""HybridFCLModel – Classical implementation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Tuple, Iterable


class HybridFCLModel(nn.Module):
    """
    Classical neural network that supports both classification and regression.
    """

    def __init__(
        self,
        num_features: int,
        task: str = "classification",
        depth: int = 1,
        hidden_size: int = 32,
    ) -> None:
        """
        Parameters
        ----------
        num_features : int
            Number of input features.
        task : str, optional
            Either ``"classification"`` or ``"regression"``. Defaults to ``"classification"``.
        depth : int, optional
            Number of hidden blocks after the initial encoding. Defaults to 1.
        hidden_size : int, optional
            Width of the hidden layers. Defaults to 32.
        """
        super().__init__()
        self.task = task

        layers: list[nn.Module] = []

        # Initial encoding – a lightweight version of the FCL layer.
        layers.append(nn.Linear(num_features, hidden_size))
        layers.append(nn.Tanh())

        # Add user‑defined depth of hidden blocks.
        for _ in range(depth):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Head
        if task == "classification":
            layers.append(nn.Linear(hidden_size, 2))
            self.activation = nn.LogSoftmax(dim=-1)
        else:
            layers.append(nn.Linear(hidden_size, 1))
            self.activation = None

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, num_features)``.
        Returns
        -------
        torch.Tensor
            Output logits (log‑softmax for classification) or scalar predictions.
        """
        out = self.net(x)
        if self.activation is not None:
            out = self.activation(out)
        return out

    @staticmethod
    def generate_superposition_data(
        num_features: int, samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data used in the QML regression seed.

        Parameters
        ----------
        num_features : int
            Number of features per sample.
        samples : int
            Number of samples to generate.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(x, y)`` where ``x`` are feature arrays and ``y`` are target values.
        """
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset that wraps the synthetic superposition data.
    """

    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = HybridFCLModel.generate_superposition_data(
            num_features, samples
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


__all__ = ["HybridFCLModel", "RegressionDataset"]
