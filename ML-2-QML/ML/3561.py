"""Hybrid classical QCNN‑style classifier that emulates the quantum circuit structure."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
from torch import nn


class HybridQCNNClassifier(nn.Module):
    """
    Classical analogue of the quantum QCNN + classifier.

    The network reproduces the layer hierarchy of the quantum ansatz:
    feature map → conv1 → pool1 → conv2 → pool2 → conv3 → head.
    All non‑linearities are ReLU/Tanh to match the variational circuit’s effective behaviour.
    """

    def __init__(self, num_features: int = 8) -> None:
        super().__init__()
        # Feature map analogous to the ZFeatureMap
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())
        # Convolutional / variational layers
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Classifier head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Standard forward pass."""
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def weight_sizes(self) -> List[int]:
        """Return the number of parameters in each layer for comparison to the quantum side."""
        sizes = []
        for layer in [
            self.feature_map,
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2,
            self.conv3,
            self.head,
        ]:
            for p in layer.parameters():
                sizes.append(p.numel())
        return sizes


def build_classifier_circuit(num_features: int, depth: int = 1) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Factory that aligns the classical classifier with the quantum circuit metadata.

    Parameters
    ----------
    num_features : int
        Number of input features (qubits in the quantum model).
    depth : int
        Unused but kept for API compatibility; the depth is baked into the fixed layers.

    Returns
    -------
    network : nn.Module
        The HybridQCNNClassifier instance.
    encoding : Iterable[int]
        Indices of the input parameters that correspond to the data‑encoding gates.
    weight_sizes : List[int]
        Number of trainable parameters per sub‑module.
    observables : List[int]
        Dummy observable indices mirroring the quantum measurement targets.
    """
    network = HybridQCNNClassifier(num_features)
    encoding = list(range(num_features))
    weight_sizes = network.weight_sizes()
    observables = list(range(2))  # placeholder for the two measurement outputs
    return network, encoding, weight_sizes, observables


__all__ = ["HybridQCNNClassifier", "build_classifier_circuit"]
