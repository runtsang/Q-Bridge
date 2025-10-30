"""
Classical hybrid QCNN + regression model.

The model mirrors the quantum QCNN architecture but replaces each
parameterised quantum layer with a classical linear block.  A
fully‑connected layer (FCL) and a sampler (softmax) are inserted
to emulate the quantum feature extraction and measurement stages.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data in the form of a superposition state
    |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.
    The target is sin(2θ) * cos(φ).
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.cos(thetas)[:, None] * np.eye(2 ** num_features)[0] + \
             np.exp(1j * phis)[:, None] * np.eye(2 ** num_features)[-1]
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the superposition data.
    """
    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QCNNGenModel(nn.Module):
    """
    Classical hybrid QCNN model.

    Architecture:
        feature_map -> conv1 -> pool1 -> conv2 -> pool2 -> conv3
        -> fcl (fully‑connected layer) -> sampler (softmax) -> head
    """
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # FCL inspired block
        self.fcl = nn.Sequential(nn.Linear(4, 2), nn.Tanh())
        # Sampler block (softmax)
        self.sampler = nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=-1))
        # Regression head
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.fcl(x)
        x = self.sampler(x)
        return torch.sigmoid(self.head(x)).squeeze(-1)


def QCNNGen() -> QCNNGenModel:
    """
    Factory returning a ready‑to‑train QCNNGenModel instance.
    """
    return QCNNGenModel()


__all__ = ["QCNNGen", "QCNNGenModel", "RegressionDataset", "generate_superposition_data"]
