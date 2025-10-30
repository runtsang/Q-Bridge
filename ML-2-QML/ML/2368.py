from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class HybridQuantumModel:
    """Hybrid classical model for classification or regression.

    The class exposes a static method ``build_classifier_circuit`` that returns a
    PyTorch network together with encoding, weight sizes and observable indices.
    The method accepts a ``task`` argument to switch between classification and
    regression behaviours.  The returned network can be used directly in a
    training loop.

    The module also provides dataset helpers that generate data compatible with the
    quantum encoding used in the QML side.
    """

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        task: str = "classification",
    ) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """
        Construct a feed‑forward network with metadata similar to the quantum variant.

        Parameters
        ----------
        num_features : int
            Number of input features / qubits.
        depth : int
            Number of hidden layers.
        task : str, optional
            Either ``"classification"`` or ``"regression"``. Determines output
            dimensionality and head activation.

        Returns
        -------
        network : nn.Module
            The constructed PyTorch network.
        encoding : List[int]
            Indices of input features used for encoding (identity mapping).
        weight_sizes : List[int]
            Number of trainable parameters per layer.
        observables : List[int]
            Dummy observable indices; for classification 0/1, for regression 0.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        if task == "classification":
            head = nn.Linear(in_dim, 2)
            layers.append(head)
            weight_sizes.append(head.weight.numel() + head.bias.numel())
            observables = [0, 1]
        else:  # regression
            head = nn.Linear(in_dim, 1)
            layers.append(head)
            weight_sizes.append(head.weight.numel() + head.bias.numel())
            observables = [0]

        network = nn.Sequential(*layers)
        return network, encoding, weight_sizes, observables

def generate_superposition_data(
    num_features: int,
    samples: int,
    task: str = "classification",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data in the form of a superposition of |0...0⟩ and |1...1⟩.

    The labels are chosen to be compatible with both classification and regression
    tasks.  For classification the labels are 0/1 based on the sign of the sum of
    the input angles; for regression the labels are a smooth sinusoidal function.

    Parameters
    ----------
    num_features : int
        Number of features / qubits.
    samples : int
        Number of data points.
    task : str, optional
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Labels of shape (samples,).
    """
    # Uniformly sample angles in [-π, π]
    angles = np.random.uniform(-np.pi, np.pi, size=(samples, num_features)).astype(np.float32)
    X = np.cos(angles).sum(axis=1) * np.sin(angles).sum(axis=1)  # simple feature
    if task == "classification":
        y = (np.sign(X) + 1) // 2  # map to {0, 1}
    else:  # regression
        y = np.sin(X) + 0.1 * np.cos(2 * X)
    return X.reshape(-1, 1), y.astype(np.float32)

class HybridDataset(Dataset):
    """
    Dataset that can produce data for either classification or regression.

    The data is generated on the fly using :func:`generate_superposition_data`.
    """

    def __init__(self, samples: int, num_features: int, task: str = "classification"):
        self.task = task
        self.samples = samples
        self.num_features = num_features
        self.features, self.labels = generate_superposition_data(num_features, samples, task)

    def __len__(self) -> int:  # type: ignore[override]
        return self.samples

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

__all__ = ["HybridQuantumModel", "generate_superposition_data", "HybridDataset"]
