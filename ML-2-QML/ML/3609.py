"""
Hybrid classical QCNN implementation.

This module defines a `HybridQCNNModel` that emulates the quantum convolution‑pooling
pipeline using fully‑connected layers and incorporates a classical approximation of
the quantum fully‑connected layer (`FCL`).  The class can be instantiated and
trained with standard PyTorch tooling.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Classical surrogate for the quantum fully‑connected layer
# --------------------------------------------------------------------------- #
class FullyConnectedLayer(nn.Module):
    """
    Mimics a quantum fully‑connected layer by applying a linear map
    followed by a tanh activation and averaging the result.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Parameters
        ----------
        thetas : Iterable[float]
            Sequence of parameters to feed into the layer.

        Returns
        -------
        np.ndarray
            One‑dimensional expectation value, compatible with the
            quantum implementation in the reference pair.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().cpu().numpy()

# --------------------------------------------------------------------------- #
# Hybrid QCNN model
# --------------------------------------------------------------------------- #
class HybridQCNNModel(nn.Module):
    """
    Classical neural network that mirrors the quantum QCNN architecture.
    The architecture consists of a feature map, three convolutional layers,
    two pooling layers, a fully‑connected layer (classical surrogate), and
    a final linear head.
    """
    def __init__(self, input_dim: int = 8, num_classes: int = 1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.fcl = FullyConnectedLayer(n_features=4)
        self.head = nn.Linear(4, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid QCNN.
        """
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Classical surrogate of the quantum fully‑connected layer
        fcl_out = torch.from_numpy(self.fcl.run(x.detach().cpu().numpy().flatten()))
        x = fcl_out
        return torch.sigmoid(self.head(x))

def HybridQCNN() -> HybridQCNNModel:
    """
    Factory function that returns a ready‑to‑train `HybridQCNNModel`.
    """
    return HybridQCNNModel()

__all__ = ["HybridQCNN", "HybridQCNNModel"]
