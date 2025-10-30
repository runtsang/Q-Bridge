"""Hybrid classical‑quantum classifier with a shared interface.

The module implements the classical side of the hybrid classifier.
It provides a feed‑forward network that mirrors the quantum API,
allowing seamless integration into joint optimisation pipelines.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data that mimics a quantum superposition.
    The labels are a non‑linear function of the feature sum to challenge the classifier.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sign(np.sin(angles))
    return x, y.astype(np.float32)

class QuantumHybridClassifier(nn.Module):
    """
    Classical feed‑forward classifier that mirrors the interface of the quantum version.
    """

    def __init__(self, num_features: int, hidden_dims: Iterable[int] = (64, 32)):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))  # binary classification logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.  The output shape is (batch, 2).
        """
        return self.net(x)

    @staticmethod
    def build_classifier_circuit(num_features: int,
                                 depth: int = 2) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Return a network, encoding indices, weight sizes and observables
        that match the quantum API.
        """
        net = QuantumHybridClassifier(num_features, hidden_dims=[64, 32])
        encoding = list(range(num_features))
        weight_sizes = [p.numel() for p in net.parameters()]
        observables = [0, 1]  # indices of the two logits
        return net, encoding, weight_sizes, observables

__all__ = ["QuantumHybridClassifier", "build_classifier_circuit", "generate_superposition_data"]
