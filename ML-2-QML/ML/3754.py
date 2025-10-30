"""Classical hybrid classifier with advanced architecture and superposition data generation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Iterable, List


def generate_superposition_data(num_features: int, samples: int, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data where each sample is a superposition of basis states
    encoded as real‑valued feature vectors.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature space.
    samples : int
        Number of data points to generate.
    threshold : float, optional
        Decision boundary for binary classification; samples with
        sum(features) > threshold are labeled 1 otherwise 0.

    Returns
    -------
    features : ndarray of shape (samples, num_features)
        Real‑valued feature matrix.
    labels : ndarray of shape (samples,)
        Binary labels.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    labels = (x.sum(axis=1) > threshold).astype(np.int64)
    return x, labels


class ClassificationDataset(Dataset):
    """
    Torch dataset that yields feature vectors and integer class labels.
    """
    def __init__(self, samples: int, num_features: int, threshold: float = 0.0):
        self.features, self.labels = generate_superposition_data(num_features, samples, threshold)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.long),
        }


class HybridClassifierML(nn.Module):
    """
    Flexible feed‑forward network for binary classification with optional residual
    connections, dropout, and layer‑norm.  The architecture is parametrised by
    ``hidden_sizes`` and ``dropout`` to allow easy scaling.

    Parameters
    ----------
    num_features : int
        Input dimensionality.
    hidden_sizes : list[int], optional
        Sizes of hidden layers; defaults to a single 64‑unit layer.
    dropout : float, optional
        Dropout probability applied after every hidden layer.
    """
    def __init__(
        self,
        num_features: int,
        hidden_sizes: List[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [64]
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(h))
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))  # binary classification
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing raw logits for two classes.
        """
        return self.net(state_batch.to(torch.float32))


def build_classifier_circuit(
    num_features: int,
    hidden_sizes: List[int] | None = None,
    dropout: float = 0.0,
) -> tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Build a classical classifier mirroring the quantum interface.

    Returns
    -------
    model : nn.Module
        The constructed network.
    encoding : Iterable[int]
        Feature indices used for the input layer.
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer.
    observables : list[int]
        Dummy observables list to keep API parity with the quantum side.
    """
    model = HybridClassifierML(num_features, hidden_sizes, dropout)
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            weight_sizes.append(layer.weight.numel() + layer.bias.numel())
    observables = [0, 1]  # placeholder for class indices
    return model, encoding, weight_sizes, observables


__all__ = ["HybridClassifierML", "ClassificationDataset", "generate_superposition_data", "build_classifier_circuit"]
