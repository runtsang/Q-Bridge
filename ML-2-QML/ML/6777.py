"""Hybrid classical-quantum inspired model for classification and regression.

This module defines a purely classical implementation that mirrors the quantum
architecture from the reference seeds.  The network consists of a CNN feature
extractor, a quantum‑style random layer emulation, and a final fully‑connected
head.  An optional regression head can be activated via the *regression*
flag.  The design demonstrates how classical layers can approximate the
behaviour of a variational quantum circuit while remaining fully
autograd‑compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data by summing angles and applying a trigonometric target.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray
        Input features of shape (samples, num_features).
    y : np.ndarray
        Target values of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data for regression."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumEmulator(nn.Module):
    """A lightweight emulation of a variational quantum layer using random weights.

    The emulator applies a sequence of linear projections followed by
    non‑linearities, mimicking the expressive power of a quantum circuit.
    """

    def __init__(self, n_ops: int = 50, out_features: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        in_features = 16 * 7 * 7  # matches the flattened output of the CNN
        for _ in range(n_ops):
            self.layers.append(nn.Linear(in_features, out_features, bias=False))
            in_features = out_features
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class HybridNATModel(nn.Module):
    """Classical hybrid model combining CNN feature extraction, a quantum‑style
    emulator, and a classification/regression head.

    Parameters
    ----------
    regression : bool, default False
        If True, the network outputs a single scalar (regression).
        Otherwise, it outputs a 4‑dimensional vector (classification).
    """

    def __init__(self, regression: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.emulator = QuantumEmulator()
        self.fc_cls = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm_cls = nn.BatchNorm1d(4)
        self.regression = regression
        if self.regression:
            self.fc_reg = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        quantum_features = self.emulator(flattened)
        if self.regression:
            out = self.fc_reg(quantum_features).squeeze(-1)
        else:
            out = self.fc_cls(quantum_features)
            out = self.norm_cls(out)
        return out


__all__ = ["HybridNATModel", "RegressionDataset", "generate_superposition_data"]
