"""Hybrid fully connected layer and regression model for classical training.

This module builds upon the original FCL and QuantumRegression seeds, merging their
classical regression dataset and fully‑connected network into a single reusable
class.  The :class:`HybridFCL` exposes both a `run` method – mimicking the simple
parameterised layer – and a standard `forward` method for arbitrary feature
vectors.  It can be trained with any PyTorch optimiser and integrated into
larger pipelines with minimal friction.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy regression dataset.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    samples : int
        Number of samples to generate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``features`` of shape ``(samples, num_features)`` and
        corresponding ``labels``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridFCL(nn.Module):
    """
    Classical hybrid fully‑connected layer.

    The network consists of a linear embedding followed by a small feed‑forward
    backbone.  It also implements a ``run`` method that accepts a list of
    rotation angles and returns the mean tanh output – a toy quantum‑style
    expectation value – mirroring the behaviour of the original FCL seed.
    """

    def __init__(self, num_features: int = 1, hidden_sizes: list[int] | None = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 16]
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Standard forward pass."""
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the quantum FCL circuit using a simple classical network.

        Parameters
        ----------
        thetas : Iterable[float]
            Rotation angles for the toy circuit.

        Returns
        -------
        np.ndarray
            Array of shape ``(1,)`` containing the mean tanh activation.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.net(values)).mean(dim=0)
        return expectation.detach().numpy()

__all__ = ["HybridFCL", "RegressionDataset", "generate_superposition_data"]
