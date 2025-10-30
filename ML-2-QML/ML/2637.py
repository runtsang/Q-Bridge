"""Hybrid classical‑graph neural network with a quantum‑enhanced embedding.

This module defines a single ``HybridQuantumNATGraph`` class that can be imported from
``QuantumNAT__gen197.py``.  The design merges the following ideas:

* A 2‑D convolutional encoder (from the first reference) that extracts local image features.
* A classical graph‑based feed‑forward network (from the second reference) that operates on the flattened feature vector.
* A quantum variational layer that encodes the graph activations into a 4‑qubit state and outputs a 4‑dimensional vector.
* A fidelity‑based adjacency graph to adaptively share parameters between different quantum sub‑circuits.
* Batch‑normalisation is applied after the quantum layer so that the classical and quantum outputs live on the same scale.

The class is fully compatible with PyTorch and can be used in the same training loops as the original QFCModel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum sub‑module
from.QuantumNAT__gen197_qml import QuantumModule, _graph_for_fidelity

# --------------------------------------------------------------------------- #
# Helper: Convolutional encoder
# --------------------------------------------------------------------------- #
class _ConvEncoder(nn.Module):
    """Two‑layer CNN that extracts local image features."""
    def __init__(self, in_channels: int = 1, out_channels: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

# --------------------------------------------------------------------------- #
# Helper: Classical graph‑based feed‑forward network
# --------------------------------------------------------------------------- #
class _GraphNetwork(nn.Module):
    """Feed‑forward network that mirrors the classical GraphQNN implementation."""
    def __init__(self, arch: list[int]) -> None:
        super().__init__()
        self.arch = arch
        self.weights = nn.ParameterList()
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            self.weights.append(nn.Parameter(torch.randn(out_f, in_f)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (bsz, in_features)
        activations = [x]
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current.T).T
            activations.append(current)
        return activations[-1]  # return last layer activation

# --------------------------------------------------------------------------- #
# Main hybrid model
# --------------------------------------------------------------------------- #
class HybridQuantumNATGraph(nn.Module):
    """Hybrid classical‑quantum model that extends QFCModel."""
    def __init__(
        self,
        conv_out_channels: int = 8,
        graph_arch: list[int] = [16, 32, 4],
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = _ConvEncoder(out_channels=conv_out_channels)
        # Compute the flattened feature size after the conv encoder
        dummy = torch.zeros(1, 1, 28, 28)
        flat_size = self.encoder(dummy).view(1, -1).shape[1]
        self._flatten_size = flat_size
        self.graph_net = _GraphNetwork(graph_arch)
        self.quantum = QuantumModule(n_qubits=n_qubits)
        self.norm = nn.BatchNorm1d(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (bsz, 1, H, W)
        bsz = x.shape[0]
        # Convolutional feature extraction
        features = self.encoder(x)
        flattened = features.view(bsz, -1)
        # Classical graph network
        graph_output = self.graph_net(flattened)  # shape: (bsz, 4)
        # Build fidelity‑based adjacency graph from the last layer activations
        adjacency = _graph_for_fidelity(graph_output)
        # Quantum layer
        q_out = self.quantum(graph_output, adjacency)
        # Normalise
        return self.norm(q_out)

__all__ = ["HybridQuantumNATGraph"]
