"""Quanvolutional filter using a parameterized quantum circuit and PennyLane."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np


class QuanvolutionFilter(nn.Module):
    """Apply a parameterized quantum circuit to 2x2 image patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.n_wires)
        # Variational parameters for a 3‑layer StronglyEntanglingLayers ansatz
        self.ansatz_params = nn.Parameter(torch.randn(3, self.n_wires, 3))
        # Optional classical post‑processing layer
        self.linear = nn.Linear(4, 4)
        # Quantum node
        self.qnode = qml.QNode(self._quantum_circuit, dev=self.dev, interface="torch", diff_method="backprop")

    def _quantum_circuit(self, patch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Quantum circuit that encodes a 4‑element patch into rotations."""
        # Encode each pixel value into a rotation around Y
        for i in range(self.n_wires):
            qml.RY(patch[i], wires=i)
        # Variational ansatz
        qml.templates.StronglyEntanglingLayers(params, wires=range(self.n_wires))
        # Measure Pauli‑Z expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, 28, 28)
        Returns:
            Tensor of shape (B, 4*14*14)
        """
        bsz = x.shape[0]
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2x2 patch and flatten to (B, 4)
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Run the quantum circuit for each batch element
                # Vectorized by stacking results
                patch_features = torch.stack(
                    [self.qnode(patch[i], self.ansatz_params) for i in range(bsz)], dim=0
                )
                # Optional classical linear layer
                patch_features = self.linear(patch_features)
                patches.append(patch_features)
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quantum filter followed by a linear head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
